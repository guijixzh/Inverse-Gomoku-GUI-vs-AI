"""构建 Cython 加速模块:python setup_fast.py build_ext --inplace"""
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="antifive-fast",
    ext_modules=cythonize(
        [Extension("antifive._fast", ["src/antifive/_fast.pyx"],
                   include_dirs=[np.get_include()])],
        language_level=3,
        quiet=True,
    ),
)
