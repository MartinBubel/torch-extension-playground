import torch
from torch import Tensor


def is_building(i: int, j: int) -> bool:
    # return torch.rand(1).item() < 0.2 if i != j else True
    return ((i + j) % 2) == 0


def derivative_rule(A2: Tensor, A1: Tensor, dx: float, i: int, j: int) -> Tensor:
    return (A2 - A1) / dx if not is_building(i, j) else A1 * 0.0


def derivative_d1(A: Tensor, dx: float) -> Tensor:
    # naive implementation
    assert A.dim() == 2, f"Expected 2D tensor, got {A.dim()}D tensor"
    rows, cols = A.shape
    derivatives = torch.zeros((rows - 1, cols))
    for i in range(rows - 1):
        for j in range(cols):
            derivatives[i][j] = derivative_rule(A[i + 1, j], A[i, j], dx, i, j)
    return derivatives


def run() -> None:
    A = torch.rand((4, 4))
    dx = 0.05
    print("A: ", A)
    derivative = derivative_d1(A, dx)
    print("derivative_d1(A): ", derivative)

    def get_derivative(A: Tensor) -> Tensor:
        return derivative_d1(A, dx)

    # print(torch.autograd.functional.jacobian(get_derivative, A))


if __name__ == "__main__":
    run()
