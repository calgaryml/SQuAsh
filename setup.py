from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mlp_hip',
    ext_modules=[
        CUDAExtension('mlp_hip', ['rocm_torch_extension/mlp.hip']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages()
    )