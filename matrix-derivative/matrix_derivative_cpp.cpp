#include <iostream>
#include <vector>
#include <torch/extension.h>

torch::Tensor matrix_derivative_forward(
    torch::Tensor values
) {
    const auto batch_size = values.size(0) - 1;
    const auto state_size = values.size(1);

    auto derivatives = torch::zeros({batch_size, state_size});

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < state_size; j++) {
            if (((i + j) % 2) == 0) {
                derivatives[i][j] = 0.;
                
            } else {
                derivatives[i][j] = (values[i + 1][j] - values[i][j]) / 0.05;
            }
        }
    }

    return derivatives;
}


torch::Tensor matrix_derivative_backward(
    torch::Tensor grad_values
) {
  throw std::runtime_error("matrix_derivative_backward is not implemented for CPP");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matrix_derivative_forward, "Matrix Derivative forward");
  m.def("backward", &matrix_derivative_backward, "Matrix Derivative backward");
}
