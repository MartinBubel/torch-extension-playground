/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>


const int sPencils = 4;  // small # pencils
const int lPencils = 32; // large # pencils

template <typename scalar_t>
__device__ __forceinline__ scalar_t derivative_rule(scalar_t a1, scalar_t a2, const int i, const int j, const scalar_t dx) {
    if (((i + j) % 2) == 0) {
        return a1 * 0.;
    } else {
        return (a2 - a1) / (2. * dx);
    }
}


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

 
// host routine to set constant data
template <size_t mx, size_t my, size_t mz>
void setDerivativeParameters(dim3 (&grid)[3][2], dim3 (&block)[3][2])
{
    // TODO: not sure if this should not also check whether mz is a multiple of sPencils and lPencils
    if ((mx % sPencils != 0) || (my %sPencils != 0) || (mz % sPencils != 0)) {
        printf("'mx', 'my', and 'mz' must be integral multiples of sPencils\n");
        exit(1);
    }
    
    if ((mx % lPencils != 0) || (my % lPencils != 0)) {
        printf("'mx' and 'my' must be multiples of lPencils\n");
        exit(1);
    }

    // Execution configurations for small and large pencil tiles

    // this is only used for testing, can be replace for non-testing setups
    grid[0][0]  = dim3(my / sPencils, mz, 1);
    block[0][0] = dim3(mx, sPencils, 1);

    grid[0][1]  = dim3(my / lPencils, mz, 1);
    block[0][1] = dim3(mx, sPencils, 1);

    grid[1][0]  = dim3(mx / sPencils, mz, 1);
    block[1][0] = dim3(sPencils, my, 1);

    grid[1][1]  = dim3(mx / lPencils, mz, 1);
    // we want to use the same number of threads as above,
    // so when we use lPencils instead of sPencils in one
    // dimension, we multiply the other by sPencils/lPencils
    block[1][1] = dim3(lPencils, my * sPencils / lPencils, 1);

    grid[2][0]  = dim3(mx / sPencils, my, 1);
    block[2][0] = dim3(sPencils, mz, 1);

    grid[2][1]  = dim3(mx / lPencils, my, 1);
    block[2][1] = dim3(lPencils, mz * sPencils / lPencils, 1);
}


template<size_t mx, size_t my, size_t mz>
void initInput(float *f, int dim)
{
    float fx = 1.0f, fy = 1.0f, fz = 1.0f;

    const float twopi = 8.f * (float)atan(1.0);

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
        for (int i = 0; i < mx; i++) {
            switch (dim) {
            case 0: 
                f[k*mx*my+j*mx+i] = cos(fx*twopi*(i-1.f)/(mx-1.f));
                break;
            case 1:
                f[k*mx*my+j*mx+i] = cos(fy*twopi*(j-1.f)/(my-1.f));
                break;
            case 2:
                f[k*mx*my+j*mx+i] = cos(fz*twopi*(k-1.f)/(mz-1.f));
                break;
            }
        }
        }
    }     
}


template <size_t mx, size_t my, size_t mz>
void initSol(float *sol, int dim)
{
    float fx = 1.0f, fy = 1.0f, fz = 1.0f;
    const float twopi = 8.f * (float)atan(1.0);

    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
        for (int i = 0; i < mx; i++) {
            switch (dim) {
            case 0: 
                sol[k*mx*my+j*mx+i] = -fx*twopi*sin(fx*twopi*(i-1.f)/(mx-1.f));
                break;
            case 1:
                sol[k*mx*my+j*mx+i] = -fy*twopi*sin(fy*twopi*(j-1.f)/(my-1.f));
                break;
            case 2:
                sol[k*mx*my+j*mx+i] = -fz*twopi*sin(fz*twopi*(k-1.f)/(mz-1.f));
                break;
            }
        }
        }
    }    
}


template <size_t mx, size_t my, size_t mz>
void checkResults(double &error, double &maxError, float *sol, float *df)
{
    // error = sqrt(sum((sol-df)**2)/(mx*my*mz))
    // maxError = maxval(abs(sol-df))
    maxError = 0;
    error = 0;
    for (int k = 0; k < mz; k++) {
        for (int j = 0; j < my; j++) {
        for (int i = 0; i < mx; i++) {
            float s = sol[k*mx*my+j*mx+i];
            float f = df[k*mx*my+j*mx+i];
            //printf("%d %d %d: %f %f\n", i, j, k, s, f);
            error += (s-f)*(s-f);
            if (fabs(s-f) > maxError) maxError = fabs(s-f);
        }
        }
    }
    error = sqrt(error / (mx*my*mz));
}


// -------------
// x derivatives
// -------------
template <size_t mx, size_t my, size_t mz>
__global__ void derivative_x(float *f, float *df)
{
    // 1 pixel-wide padding on each side
    // TODO: this should be a parameter
    __shared__ float s_f[sPencils][mx+2]; // 1-wide halo

    const float dx = 1. / (mx-1.f);

    int i = threadIdx.x;
    int j = blockIdx.x*blockDim.y + threadIdx.y;
    int k = blockIdx.y;
    int si = i + 4;       // local i for shared memory access + halo offset
    int sj = threadIdx.y; // local j for shared memory access

    int globalIdx = k * mx * my + j * mx + i;

    s_f[sj][si] = f[globalIdx];

    __syncthreads();

    // fill in periodic images in shared memory array
    // I updated this as I am using a much simpler derivative rule here
    if (i < 1) {
        s_f[sj][si-1]  = s_f[sj][si+mx-2];
        s_f[sj][si+mx] = s_f[sj][si+1];   
    }

    __syncthreads();
    
    df[globalIdx] = derivative_rule(s_f[sj][si+1], s_f[sj][si-1], i, j, dx);
}

// Run the kernels for a given dimension. One for sPencils, one for lPencils
template <size_t mx, size_t my, size_t mz>
void runTest(int dimension, dim3 (&grid)[3][2], dim3 (&block)[3][2])
{
    void (*fpDeriv[2])(float*, float*);
    fpDeriv[0] = derivative_x<mx, my, mz>;

    int sharedDims[3][2][2] = {
        mx,
        sPencils,
        mx,
        lPencils,
        sPencils,
        my,
        lPencils,
        my,
        sPencils,
        mz,
        lPencils,
        mz
    };

  float *f = new float[mx*my*mz];
  float *df = new float[mx*my*mz];
  float *sol = new float[mx*my*mz];                           
    
  initInput<mx, my, mz>(f, dimension);
  initSol<mx, my, mz>(sol, dimension);

  // device arrays
  int bytes = mx*my*mz * sizeof(float);
  float *d_f, *d_df;
  checkCuda( cudaMalloc((void**)&d_f, bytes) );
  checkCuda( cudaMalloc((void**)&d_df, bytes) );

  const int nReps = 20;
  float milliseconds;
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  double error, maxError;

  printf("%c derivatives\n\n", (char)(0x58 + dimension));

  for (int fp = 0; fp < 2; fp++) { 
    checkCuda( cudaMemcpy(d_f, f, bytes, cudaMemcpyHostToDevice) );  
    checkCuda( cudaMemset(d_df, 0, bytes) );
    
    fpDeriv[fp]<<<grid[dimension][fp],block[dimension][fp]>>>(d_f, d_df); // warm up
    checkCuda( cudaEventRecord(startEvent, 0) );
    for (int i = 0; i < nReps; i++)
       fpDeriv[fp]<<<grid[dimension][fp],block[dimension][fp]>>>(d_f, d_df);
    
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

    checkCuda( cudaMemcpy(df, d_df, bytes, cudaMemcpyDeviceToHost) );
        
    checkResults<mx, my, mz>(error, maxError, sol, df);

    printf("  Using shared memory tile of %d x %d\n", 
           sharedDims[dimension][fp][0], sharedDims[dimension][fp][1]);
    printf("   RMS error: %e\n", error);
    printf("   MAX error: %e\n", maxError);
    printf("   Average time (ms): %f\n", milliseconds / nReps);
    printf("   Average Bandwidth (GB/s): %f\n\n", 
           2.f * 1e-6 * mx * my * mz * nReps * sizeof(float) / milliseconds);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );

  checkCuda( cudaFree(d_f) );
  checkCuda( cudaFree(d_df) );

  delete [] f;
  delete [] df;
  delete [] sol;
}


torch::Tensor core_matrix_derivative_cuda_forward(torch::Tensor values) {
    const auto mx = 64;  // values.size(0);
    const auto my = 64;  // values.size(1);
    const auto mz = 64;  // values.size(2);
    
    // TODO: not sure if the below holds the type as the below-below
    // auto derivatives = torch::ones_like(values);
    auto derivatives = torch::ones({mx, my, mz}, values.type());

    auto grid = dim3(my / sPencils, mz, 1);
    auto block = dim3(mx, sPencils, 1);

    // derivative_x<mx, my, mz><<<grid, block>>>(values.data<float>(), derivatives.data<float>());
    derivative_x<mx, my, mz><<<grid, block>>>(values.data_ptr<float>(), derivatives.data_ptr<float>());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    return derivatives;
}


// This the main host code for the finite difference 
// example.  The kernels are contained in the derivative_m module

int main(void)
{
    // Print device and precision
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    const size_t mx = 64;
    const size_t my = 64;
    const size_t mz = 64;

    dim3 grid[3][2], block[3][2];
    std::cout << "grid: " << grid << std::endl;
    std::cout << "block: " << block << std::endl;

    setDerivativeParameters<mx, my, mz>(grid, block); // initialize 

    runTest<mx, my, mz>(0, grid, block); // x derivative

    return 0;
}