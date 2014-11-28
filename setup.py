import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = "Verification distance function",
    cmdclass = {"build_ext": build_ext},
    packages = ['verification'],
    ext_modules = [Extension("verification/distances", ["verification/distances.pyx"],
                             include_dirs = [np.get_include()]),
                   Extension("verification/distances_sparse", ["verification/distances_sparse.pyx"],
                             include_dirs = [np.get_include()])],
    install_requires=['numpy', 'scikit-learn'],
)

# compile this via: python setup.py build_ext --inplace
