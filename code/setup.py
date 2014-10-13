from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
	name = "Min-Max distance function",
	cmdclass = {"build_ext": build_ext},
    ext_modules = [Extension("minmax", ["minmax.pyx"], include_dirs = [np.get_include()])]#["Users/mike/anaconda/lib/python3.4/site-packages/numpy/core/include"])]
)