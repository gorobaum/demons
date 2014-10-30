from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("callDemon", ["callDemonmodule.cxx"])

setup(name = "callDemon", ext_modules=[extension_mod])
