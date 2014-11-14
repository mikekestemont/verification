import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
	name = "Verification distance function",
	cmdclass = {"build_ext": build_ext},
    ext_modules = [Extension("distances", ["distances.pyx"], include_dirs = [np.get_include()])],
    install_requires=['numpy', 'scikit-learn', 'PLM'],
    dependency_links=[
       "https://github.com/fbkarsdorp/PLM/archive/master.zip#egg=PLM-0.0.2"]
)

# compile this via: python setup.py build_ext --inplace
