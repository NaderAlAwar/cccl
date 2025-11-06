# S5 2D Scan Implementation Attempts with cuda.compute

This directory contains various attempts to implement the 2D S5 operator using a **single scan call** with `cuda.compute`, mirroring the C++ implementation in `s5_operator.cu`.

## Running the Demos

Each attempt file can be run directly to demonstrate the approach and see where it fails:

```bash
# Attempt 1: TransformIterator with array return (FAILS)
python attempt1_transform_iterator.py

# Attempt 2: Pointer arithmetic approach (FAILS - output iterator issues)
python attempt2_pointer_arithmetic.py

# Attempt 3: Custom DualRowPointerIterator (WORKS! ✓)
python attempt3_pointer_cast.py
compute-sanitizer --tool memcheck python attempt3_pointer_cast.py  # Verify with memcheck
```

## Goal

Implement 2D S5 scan where:
- Input: `A` and `Bu` matrices with shape `(timesteps, state_dim)`
- Operation: Scan over timesteps with each "element" being a full row vector
- Single scan call over `timesteps` iterations (not `timesteps * state_dim`)
- Operator performs elementwise operations on row pairs:
  ```
  result.A[i] = y.A[i] * x.A[i]  for all i in state_dim
  result.Bu[i] = y.A[i] * x.Bu[i] + y.Bu[i]  for all i in state_dim
  ```

## Attempts

### Attempt 1: TransformIterator with Row Extraction (`attempt1_transform_iterator.py`)

**Approach:**
```
CountingIterator(0, 1, 2, ..., timesteps-1)
  → TransformIterator(idx → array[idx])  ❌ FAILS HERE
  → ZipIterator(A_rows, Bu_rows)
  → Scan with row-wise operator
```

**Why it fails:**
- NumPy CUDA device functions cannot return arrays unless passed as arguments
- Error: `"Only accept returning of array passed into the function as argument"`
- The transform function `get_row(idx) -> d_array[idx]` tries to return an array

**Status:** ❌ Does not work

---

### Attempt 2: Pointer Arithmetic (`attempt2_pointer_arithmetic.py`)

**Approach:**
```
CountingIterator(0, 1, 2, ...)
  → TransformIterator(idx → uint64_pointer)  ✓ This works!
  → ZipIterator(A_ptrs, Bu_ptrs)
  → Scan with operator that dereferences pointers  ❌ FAILS HERE
```

**Why it partially works:**
- Transform functions return `uint64` pointers (scalars), not arrays
- Pointer arithmetic compiles: `base_ptr + idx * stride`
- Successfully creates iterator over pointer pairs

**Why it still fails:**
- The operator receives pointer pairs `(uint64, uint64)`
- Must dereference pointers to load row data
- Dereferencing requires returning/working with arrays in device code
- Same fundamental numba limitation applies in the operator

**Status:** ❌ Does not work (different failure point, same root cause)

---

### Attempt 3: Custom DualRowPointerIterator with CPointers (`attempt3_pointer_cast.py`)

**Approach:**
```
DualRowPointerIterator
  - Stores CPointer(float32) for both A and Bu in state
  - input_dereference: Computes row pointers, loads values element-by-element
  - output_dereference: Computes row pointers, writes values element-by-element
  - Returns UniTuple(UniTuple(float32, 4), 2) - value tuples, not pointers!

→ Operator works on value tuples (all scalars)
→ Single scan over timesteps
```

**Key insights that made it work:**

1. **Use CPointers from the start** - Store as `CPointer(float32)` in state, not `uint64`
2. **Pointer arithmetic semantics** - `CPointer + N` advances by N *elements*, not bytes
   - Correct: `ptr + row_idx * (row_stride_bytes // 4)`
   - Wrong: `ptr + row_idx * row_stride_bytes` (off by 4x!)
3. **Load values inside iterator** - Don't return pointers, return loaded values
4. **Element-by-element operations** - All `ptr[i]` operations return scalars
5. **No TransformIterator needed** - Built loading/storing directly into custom iterator

**Status:** ✅ **WORKS!** Achieves single scan call for 2D S5 operation

**Verification:**
- Passes memcheck with 0 errors
- Matches PyTorch reference exactly (max error: 0.000000e+00)
- Single scan call over timesteps (not timesteps × state_dim)

---

## Root Cause Analysis

The fundamental issue is **numba CUDA's restriction on array returns**:

```python
@cuda.jit(device=True)
def some_function(idx):
    return array[idx]  # ❌ NOT ALLOWED
```

This restriction exists because:
1. Device functions need predictable stack sizes
2. Returning arrays by value is expensive
3. Array lifetime management is complex on device

### Why C++ Works

The C++ equivalent successfully uses:

1. **Pointers, not values:**
   ```cpp
   IndexToPointerFunctor: idx → T* (returns pointer)
   ```

2. **VectorPair with explicit construction:**
   ```cpp
   VectorPair(const T* a_ptr, const T* bu_ptr) {
       for (int i = 0; i < DIM; i++) {
           A[i] = a_ptr[i];
           Bu[i] = bu_ptr[i];
       }
   }
   ```

3. **Flexible iterator value types:**
   - Thrust iterators can have struct value types
   - Structs can contain arrays: `struct VectorPair { T A[40]; T Bu[40]; }`
   - Construction happens in-place during dereference

## Potential Solutions

### Solution 1: Custom Strided Iterator (Most Promising)

Implement a custom iterator similar to `strided_iterator_reference/_strided.py` but modified for 2D row access:

```python
class RowIterator(IteratorBase):
    """Iterator that dereferences to row pointers or indices"""

    @staticmethod
    def dereference(state_ref):
        # Handle 2D indexing internally
        # Return pointer or use different approach
        pass
```

**Challenge:** Current strided iterator returns scalars. Need to extend for vector values or use column-wise iteration with multiple scans.

### Solution 2: Infrastructure Changes

Modify `cuda.compute` to support:
- Struct value types with array fields
- Pointer value types with automatic dereferencing
- Special handling for 2D array iteration

**Challenge:** Requires significant changes to cuda.compute internals.

### Solution 3: Column-wise Scan Loop

Instead of single scan over rows, run multiple scans over columns:

```python
for col in range(state_dim):
    # Strided iterator for A[:, col] and Bu[:, col]
    # Single scan over this column
    # Store results in output[:, col]
```

**Challenge:** Not a single scan call, but `state_dim` separate scans.

### Solution 4: Flatten and Reshape (Naive)

Treat 2D arrays as 1D and scan over all elements:

```python
# Scan over timesteps * state_dim elements
# But S5 operator needs row structure...
```

**Challenge:** S5 operator semantics require row-level operations. Pure flattening doesn't preserve the necessary structure.

## Comparison with PyTorch

PyTorch's `associative_scan` handles this elegantly because:
1. Works at a higher abstraction level (eager mode)
2. Operator can manipulate tensors directly (not in compiled kernel)
3. Compilation happens after operator is defined with tensor types

```python
# PyTorch - works naturally
torch.associative_scan(
    lambda x, y: (y[0] * x[0], y[0] * x[1] + y[1]),
    (A_2d, Bu_2d),
    dim=0  # Scan over first dimension
)
```

## Summary: Working Solution

**Attempt 3 successfully implements single-scan 2D S5 operation!**

**Architecture:**
- Custom `DualRowPointerIterator` that handles both A and Bu matrices
- Stores `CPointer(float32)` in iterator state (not uint64 integers)
- Computes row pointers using proper element-wise arithmetic
- Loads/stores values element-by-element (all scalar operations)
- Returns/receives value tuples, not pointers

**Performance:**
- Single scan call over `timesteps` iterations
- Each iteration processes full vector pair (state_dim elements)
- Matches C++ VectorPair pattern conceptually
- Verified correct with compute-sanitizer memcheck

**Required Infrastructure Fixes:**
- `_iterators.py`: Fixed `underlying_it_type` scoping for output iterators
- `_zip_iterator.py`: Allow partial output support in ZipIterator

**Limitations:**
- Hardcoded for `state_dim=4` (requires manual unrolling)
- More complex than column-wise approach
- Requires deep understanding of numba type system

## Next Steps

1. **Generalize for arbitrary state_dim** - Use code generation for loop unrolling
2. **Performance comparison** - Benchmark vs PyTorch and column-wise approach
3. **Upstream infrastructure fixes** - Submit fixes to cuda.compute
4. **Documentation** - Add as example/tutorial for custom iterators

## References

- C++ implementation: `../s5_operator.cu`
- Strided iterator: `../strided_iterator_reference/_strided.py`
- PyTorch associative_scan: Used in reference implementation
- Working solution: `attempt3_pointer_cast.py`
