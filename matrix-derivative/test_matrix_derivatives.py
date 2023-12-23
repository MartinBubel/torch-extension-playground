import torch


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
