from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("process_snv_mat.pyx", annotate=True, language_level=3)
)