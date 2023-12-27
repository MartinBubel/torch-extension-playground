from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


setup(
    name="matrix_derivative_cpp",
    ext_modules=[
        CppExtension("matrix_derivative_cpp", ["matrix_derivative_cpp.cpp"]),
        CUDAExtension(
            "matrix_derivative_cuda",
            ["matrix_derivative_cuda.cpp", "matrix_derivative_cuda_kernel.cu"],
        ),
        CUDAExtension(
            "core_matrix_derivative_cuda",
            [
                "core_matrix_derivative.cpp",
                "core_matrix_derivative_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
