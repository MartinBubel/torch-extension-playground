#include <iostream>
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor matrix_derivative_cuda_forward(torch::Tensor values);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matrix_derivative_forward(torch::Tensor values) {
  CHECK_INPUT(values);
  auto output = matrix_derivative_cuda_forward(values);
  std::cout << "output: " << output << std::endl;
  return output;
}

torch::Tensor matrix_derivative_backward(
    torch::Tensor grad_values
) {
  CHECK_INPUT(grad_values);

  throw std::runtime_error("matrix_derivative_backward is not implemented for CUDA");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matrix_derivative_forward, "Matrix derivative forward (CUDA)");
  m.def("backward", &matrix_derivative_backward, "Matrix derivative backward (CUDA)");
}
