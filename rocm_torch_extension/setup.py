from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mlp_hip',
    ext_modules=[
        CUDAExtension('mlp_hip', ['mlp.hip']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })