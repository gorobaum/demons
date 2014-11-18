from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("callDemon", ["callDemonmodule.cxx", "demonsfunction.cxx", "symmetricdemonsfunction.cxx", "vectorfield.cxx", "asymmetricdemonsfunction.cxx", "interpolation.cxx"], extra_compile_args=['-std=c++11','-fopenmp', '-O3'], extra_link_args=['-lgomp'])

setup(name = "callDemon", ext_modules=[extension_mod])
