from distutils.core import setup, Extension
import numpy

mod = Extension('CorrCoef',
    include_dirs = [numpy.get_include()],
    sources = ['CorrCoef.c'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp']
)

setup (name = 'CorrCoef',
    author = 'Aljoscha Rheinwalt',
    author_email = 'aljoscha.rheinwalt@uni-potsdam.de',
    ext_modules = [mod]
)
