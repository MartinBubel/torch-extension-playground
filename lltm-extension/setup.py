from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


setup(
    name="lltm_cpp",
    ext_modules=[
        CppExtension("lltm_cpp", ["lltm.cpp"]),
        CUDAExtension("lltm_cuda", ["lltm_cuda.cpp", "lltm_cuda_kernel.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
