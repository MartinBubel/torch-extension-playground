#include <iostream>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template <typename scalar_t>
__device__ __forceinline__ scalar_t derivative_rule(scalar_t a1, scalar_t a2, const int i, const int j) {
    if (((i + j) % 2) == 0) {
        return a1 * 0.;
    } else {
        return (a2 - a1) / 0.05;
    }
}


template <typename scalar_t>
__global__ void matrix_derivative_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> derivatives
) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;

    // printf("BlockIdx.x: %d, ThreadIdx.x: %d\n", blockIdx.x, threadIdx.x);
    // printf("values[%d][%d] = %f\n", i, j, values[i][j]);
    // printf("derivatives[%d][%d] = %f\n", i, j, derivatives[i][j]);

    derivatives[i][j] = derivative_rule(values[i][j], values[i + 1][j], i, j);
}

torch::Tensor matrix_derivative_cuda_forward(torch::Tensor values) {
    const auto batch_size = values.size(0) - 1;
    const auto state_size = values.size(1);

    // omitting the type caused A LOT OF PROBLEMS!!!!
    auto derivatives = torch::ones({batch_size, state_size}, values.type());

    const int blocks = batch_size;
    const int threads = state_size;

    AT_DISPATCH_FLOATING_TYPES(values.type(), "matrix_derivative_cuda_forward_kernel", ([&] {
        matrix_derivative_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            derivatives.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    // it might be advantageous to be calling this from python as then, an actual error is raised
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    return derivatives;
}