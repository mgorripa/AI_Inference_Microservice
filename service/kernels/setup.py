from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "kernels.binding",
        ["binding.cpp"],
        define_macros=[("CPU_ONLY", "1")],
        cxx_std=17,
    ),
]

setup(
    name="kernels",
    version="0.0.1",
    packages=["kernels"],
    package_dir={"kernels": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
