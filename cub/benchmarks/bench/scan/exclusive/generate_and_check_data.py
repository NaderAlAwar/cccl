import argparse

import numpy as np
import torch

BUILD_DIR = "/home/coder/cccl/build/cuda12.9-gcc14/cub-cpp20"
STATE_DIM = 40


def gerate_npy_files():
    A_in = np.random.randn(100000, STATE_DIM).astype(np.float32)
    Bu_in = np.random.randn(100000, STATE_DIM).astype(np.float32)

    # A_in = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    # Bu_in = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)

    with open(f"{BUILD_DIR}/A_in.npy", "wb") as f:
        np.save(f, A_in)
    with open(f"{BUILD_DIR}/Bu_in.npy", "wb") as f:
        np.save(f, Bu_in)


def load_results_and_check():
    A_in = torch.from_numpy(np.load(f"{BUILD_DIR}/A_in.npy")).cuda()
    Bu_in = torch.from_numpy(np.load(f"{BUILD_DIR}/Bu_in.npy")).cuda()

    def s5_operator(x, y):
        A_i, Bu_i = x
        A_j, Bu_j = y
        return (A_j * A_i, A_j * Bu_i + Bu_j)

    A_out, Bu_out = torch._higher_order_ops.associative_scan(
        combine_fn=s5_operator,
        xs=(A_in, Bu_in),
        dim=0,
        reverse=False,
        combine_mode="pointwise",
    )

    A_out_cpp_s5 = torch.from_numpy(np.load(f"{BUILD_DIR}/A_out_cpp_s5.npy")).cuda()
    Bu_out_cpp_s5 = torch.from_numpy(np.load(f"{BUILD_DIR}/Bu_out_cpp_s5.npy")).cuda()

    A_out_cpp_s5_segmented = torch.from_numpy(
        np.load(f"{BUILD_DIR}/A_out_cpp_s5_segmented.npy")
    ).cuda()
    Bu_out_cpp_s5_segmented = torch.from_numpy(
        np.load(f"{BUILD_DIR}/Bu_out_cpp_s5_segmented.npy")
    ).cuda()

    # print("Outputs from PyTorch associative_scan:")
    # print(A_out)
    # print(Bu_out)

    # print("Outputs from C++ verify_s5:")
    # print(A_out_cpp_s5)
    # print(Bu_out_cpp_s5)

    # print("Outputs from C++ verify_s5_segmented:")
    # print(A_out_cpp_s5_segmented)
    # print(Bu_out_cpp_s5_segmented)

    assert torch.allclose(A_out, A_out_cpp_s5, atol=1e-5), (
        "A_out does not match A_out_cpp_s5"
    )
    assert torch.allclose(Bu_out, Bu_out_cpp_s5, atol=1e-5), (
        "Bu_out does not match Bu_out_cpp_s5"
    )
    assert torch.allclose(A_out, A_out_cpp_s5_segmented, atol=1e-5), (
        "A_out does not match A_out_cpp_s5_segmented"
    )
    assert torch.allclose(Bu_out, Bu_out_cpp_s5_segmented, atol=1e-5), (
        "Bu_out does not match Bu_out_cpp_s5_segmented"
    )

    print("All results match successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and check npy files")
    parser.add_argument("--generate", action="store_true", help="Generate npy files")
    parser.add_argument("--check", action="store_true", help="Load results and check")

    args = parser.parse_args()

    if args.generate:
        gerate_npy_files()
    if args.check:
        load_results_and_check()
