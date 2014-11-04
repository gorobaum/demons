from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("callDemon", ["callDemonmodule.cxx"], extra_compile_args=['-std=c++11'])

setup(name = "callDemon", ext_modules=[extension_mod])
