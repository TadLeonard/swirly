import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import swirl


ext_module = Extension(
    "swirlop",
    ["swirlop.pyx"],
#    extra_compile_args=['-fopenmp'],
#    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()],
)


setup(
    name='swirl',
    version=swirl.__version__,
    py_modules=["swirl"],
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
)
