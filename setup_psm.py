from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
    Extension(
        "process_snv_mat",
        ["process_snv_mat.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules = cythonize(ext_modules, annotate=True, language_level=3)
)