import torch
from matrix_derivative_python import derivative_d1
from matrix_derivative import set_cpu_backend, MatrixDerivative, set_cuda_backend


print("--------------------------------------------------------------")
print("--------------------------------------------------------------")

A = torch.rand((3, 3))
print("A: ", A)

print("Python:\n", derivative_d1(A, 0.05))
print("--------------------------------------------------------------")

set_cpu_backend()
device = torch.device("cpu")
module = MatrixDerivative().to(device)
A = A.to(device)
print("CPU:\n", module(A))
print("--------------------------------------------------------------")

assert torch.cuda.is_available()
set_cuda_backend()
device = torch.device("cuda")
module = MatrixDerivative().to(device)
A = A.to(device)
output = module(A)
# torch.cuda.synchronize()
print("CUDA:\n", output)

print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
