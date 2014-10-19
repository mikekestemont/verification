from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
	name = "Verification distance function",
	cmdclass = {"build_ext": build_ext},
    ext_modules = [Extension("distances", ["distances.pyx"], include_dirs = [np.get_include()])]
)

# compile this via: python setup.py build_ext --inplace