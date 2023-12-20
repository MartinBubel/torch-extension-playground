import time
import torch


def benchmark(version, lltm_type, device, sync) -> None:
    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features, device=device)
    h = torch.randn(batch_size, state_size, device=device)
    C = torch.randn(batch_size, state_size, device=device)

    rnn = lltm_type(input_features, state_size).to(device)

    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))

        if sync:
            torch.cuda.synchronize()

        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()

        if sync:
            torch.cuda.synchronize()

        backward += time.time() - start

    print(
        "{} | Forward: {:.3f} s | Backward {:.3f} s".format(version, forward, backward)
    )


if __name__ == "__main__":
    from lltm import LLTM as LLTM_cpp
    from lltm_native_python import LLTM as LLTM_native_python

    # from torch.utils.cpp_extension import load

    # lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"])

    cpu_device = torch.device("cpu")  # device object representing CPU

    # benchmark("CPU: Python", LLTM_native_python, cpu_device, False)
    # benchmark("CPU: C++", LLTM_cpp, cpu_device, False)
    # benchmark("CPU: JIT C++", lltm_cpp.LLTM, cpu_device, False)

    print("-------------------------------")

    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    benchmark("GPU: Python", LLTM_native_python, cuda_device, True)
    benchmark("GPU: C++", LLTM_cpp, cuda_device, True)
    # benchmark("GPU: JIT C++", lltm_cpp.LLTM, cuda_device, True)
