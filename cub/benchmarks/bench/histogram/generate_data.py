import numpy as np
import torch


def generate_input(size, contention, seed):
    """
    Generates random input tensor for histogram.

    Args:
        size: Size of the input tensor (must be multiple of 16)
        contention: float in [0, 100], specifying the percentage of identical values
        seed: Random seed
    Returns:
        The input tensor with values in [0, 255]
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    # Generate integer values between 0 and 256
    data = torch.randint(
        0, 256, (size,), device="cuda", dtype=torch.uint8, generator=gen
    )

    # make one value appear quite often, increasing the chance for atomic contention
    evil_value = torch.randint(
        0, 256, (), device="cuda", dtype=torch.uint8, generator=gen
    )
    evil_loc = torch.rand(
        (size,), device="cuda", dtype=torch.float32, generator=gen
    ) < (contention / 100.0)
    data[evil_loc] = evil_value

    return data.contiguous()


size = 10485760
contention = 10
seed = 42

input = generate_input(size, contention, seed)
np.save("data.npy", input.cpu().numpy())
