import time
import torch


def bench(fun, data, num_iters=100):
    start_time = time.time()
    for _ in range(num_iters):
        fun()
    end_time = time.time()
    run_time = end_time - start_time

    bandwidth = data.numel() * data.element_size() * num_iters / run_time / 1e9

    return bandwidth


class TestMatrixDerivative:
    def _setup(self) -> None:
        self.A = torch.rand((400, 400))

    def test_python(self, benchmark) -> None:
        from matrix_derivative_python import derivative_d1

        self._setup()

        def fun():
            return derivative_d1(self.A, 0.05)

        benchmark(fun)

        assert True

    def test_cpp(self, benchmark) -> None:
        from matrix_derivative import set_cpu_backend, MatrixDerivative

        self._setup()
        set_cpu_backend()
        device = torch.device("cpu")

        module = MatrixDerivative().to(device)
        A = self.A.to(device)

        def fun():
            return module(A)

        benchmark(fun)

        assert True

    def test_cuda(self, benchmark) -> None:
        from matrix_derivative import set_cuda_backend, MatrixDerivative

        assert torch.cuda.is_available()
        device = torch.device("cuda")

        self._setup()
        set_cuda_backend()

        module = MatrixDerivative().to(device)
        A = self.A.to(device)

        def fun():
            return module(A)

        benchmark(fun)

        assert True

    def test_core_cuda(self, benchmark) -> None:
        from matrix_derivative import set_core_cuda_backend, MatrixDerivative

        assert torch.cuda.is_available()
        device = torch.device("cuda")

        # self._setup()
        self.A = torch.rand([640] * 3)  # using a 3D tensor here for simplicity
        set_core_cuda_backend()

        module = MatrixDerivative().to(device)
        A = self.A.to(device)

        def fun():
            return module(A)

        benchmark(fun)

        assert True


if __name__ == "__main__":
    # python
    from matrix_derivative_python import derivative_d1

    A = torch.rand((64, 64))

    def fun():
        return derivative_d1(A, 0.05)

    print(f"Python bandwitdh: {bench(fun, A)}")

    # CPP
    from matrix_derivative import set_cpu_backend, MatrixDerivative

    set_cpu_backend()
    device = torch.device("cpu")
    A = A.to(device)
    module = MatrixDerivative().to(device)

    def fun():
        return module(A)

    print(f"CPP bandwitdh: {bench(fun, A)}")

    # cuda
    from matrix_derivative import set_cuda_backend, MatrixDerivative

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    A = A.to(device)
    set_cuda_backend()
    module = MatrixDerivative().to(device)

    def fun():
        return module(A)

    print(f"Cuda bandwitdh: {bench(fun, A)}")

    # core cuda
    from matrix_derivative import set_core_cuda_backend, MatrixDerivative

    assert torch.cuda.is_available()
    A = torch.rand([640] * 3).to(device)
    set_core_cuda_backend()
    module = MatrixDerivative().to(device)

    def fun():
        return module(A)

    print(f"Core cuda bandwitdh: {bench(fun, A)}")
