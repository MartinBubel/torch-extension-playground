#include <iostream>
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor core_matrix_derivative_cuda_forward(torch::Tensor values);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor core_matrix_derivative_forward(torch::Tensor values) {
  CHECK_INPUT(values);
  auto output = core_matrix_derivative_cuda_forward(values);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &core_matrix_derivative_forward, "Core matrix derivative forward (CUDA)");
}
