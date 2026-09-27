from glob import glob
import os

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            options = ["/O2", "/W0", "/std:c++17"]
        else:
            options = ["-O3", "-std=c++17"]
        for extension in self.extensions:
            extension.extra_compile_args = options
        super().build_extensions()


extension = Extension(
    "pykeyatm._core.keyATM_scr",
    sources=glob(os.path.join("cpp", "*.cpp")),
    include_dirs=[
        pybind11.get_include(),
        os.path.abspath(os.path.join("cpp", "vendor", "eigen")),
    ],
    language="c++",
)

setup(ext_modules=[extension], cmdclass={"build_ext": BuildExt})
