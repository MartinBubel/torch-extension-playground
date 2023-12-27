import math
import torch


_BACKEND = None


def set_cpu_backend() -> None:
    import matrix_derivative_cpp

    global _BACKEND
    _BACKEND = matrix_derivative_cpp


def set_cuda_backend() -> None:
    import matrix_derivative_cuda

    global _BACKEND
    _BACKEND = matrix_derivative_cuda


def set_core_cuda_backend() -> None:
    import core_matrix_derivative_cuda

    global _BACKEND
    _BACKEND = core_matrix_derivative_cuda


class MatrixDerivativeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        # assert values.dim() == 2, f"Input must be a 2D matrix (got {values.dim()}D)"
        assert _BACKEND is not None, "Backend not set"
        outputs = _BACKEND.forward(values)
        ctx.save_for_backward(*outputs)
        return outputs

    @staticmethod
    def backward(ctx, values):
        assert _BACKEND is not None, "Backend not set"
        outputs = _BACKEND.backward(values.contiguous())
        return outputs


class MatrixDerivative(torch.nn.Module):
    def __init__(self):
        super(MatrixDerivative, self).__init__()

    def forward(self, values):
        # assert values.dim() == 2, f"Input must be a 2D matrix (got {values.dim()}D)"
        return MatrixDerivativeFunction.apply(values)
