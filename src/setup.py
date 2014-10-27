from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("callDemon", ["callDemonmodule.cxx", "calldemon.cxx"])

setup(name = "callDemon", ext_modules=[extension_mod])
