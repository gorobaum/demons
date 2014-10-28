#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cstdlib>
#include "calldemon.h"

static PyObject *
callDemon(PyObject *self, PyObject *args)
{
    PyObject * image = NULL;
    PyObject * dataAux = NULL;
    PyArrayObject * array = NULL;

    if (!PyArg_ParseTuple(args, "O", &image)) return NULL;

    dataAux = PyObject_CallMethod(image, "get_data", NULL);

    array = (PyArrayObject*)PyArray_FROM_OTF(dataAux, 2, 0);

    std::cout << "Dimensoes = " << PyArray_NDIM(array) << "\n";
    std::cout << "Tipo = " << PyArray_DESCR(array)->kind << "\n";
    unsigned char * macaco = static_cast<unsigned char*>(PyArray_GETPTR3(array, 0, 0, 0));
    std::cout << "[0][0][0] = " << (int)*macaco << "\n";

    Py_DECREF(array);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef CallDemonMethods[] = {
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

