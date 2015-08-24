import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from swirly.version import __version__


ext_module = Extension(
    "swirly/_swirlop",
    ["swirly/_swirlop.pyx"],
    include_dirs=[numpy.get_include()],
    #extra_compile_args=["-fopenmp"],
    #extra_link_args=["-fopenmp"],
)


setup(
    name="swirl",
    version=__version__,
    scripts=["swirl.py"], 
    packages=["swirly"],
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext_module],
)
