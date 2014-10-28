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

    std::cout << "Tipo = " << PyArray_DTYPE(array) << "\n";
    std::cout << "É continuo = " << PyArray_ISCONTIGUOUS(array) << "\n";
    std::cout << "[0][0][0] = " << static_cast<int*>(PyArray_GETPTR3(array, 0, 0, 0)) << "\n";

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

