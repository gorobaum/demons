#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "image.h"

static void
loadImageData(PyArrayObject * data, Image<unsigned char>& image) {
    std::vector<int> dims = image.getDims();
    for (int i = 0; i < dims[0]; i++)
        for (int j = 0; j < dims[1]; j++)
            for (int k = 0; k < dims[2]; k++) {
                unsigned char macaco = *static_cast<unsigned char*>(PyArray_GETPTR3(data, i, j, k));
                image(i, j, k) = macaco;
            }
}

extern "C"
PyObject *
callDemon(PyObject *self, PyObject *args) {
    PyObject * image = NULL;
    PyObject * dataAux = NULL;
    PyArrayObject * array = NULL;

    if (!PyArg_ParseTuple(args, "O", &image)) return NULL;
    dataAux = PyObject_CallMethod(image, "get_data", NULL);
    array = (PyArrayObject*)PyArray_FROM_OTF(dataAux, 2, 0);

    std::vector<int> dims;
    dims.push_back(PyArray_SHAPE(array)[0]);
    dims.push_back(PyArray_SHAPE(array)[1]);
    dims.push_back(PyArray_SHAPE(array)[2]);

    Image<unsigned char> imageC(dims);

    loadImageData(array, imageC);

    std::cout << "Dae = " << (int)imageC(0,0,0) << "\n";

    Py_DECREF(array);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef 
CallDemonMethods[] = {
    {"calldemon",  callDemon, METH_VARARGS,
     "Call the demons registration method."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initcallDemon(void)
{
    (void) Py_InitModule("callDemon", CallDemonMethods);
    import_array();
}

