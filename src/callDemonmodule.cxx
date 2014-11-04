#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "image.h"
#include "demons.h"

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

PyArrayObject * transformToArray(PyObject * image) {
    PyObject * data = PyObject_CallMethod(image, "get_data", NULL);
    return (PyArrayObject*)PyArray_FROM_OTF(data, 2, 0);
}

extern "C"
PyObject *
callDemon(PyObject *self, PyObject *args) {
    PyObject * staticImagePy = NULL;
    PyObject * movingImagePy = NULL;
    PyArrayObject * staticImageDataArray = NULL;
    PyArrayObject * movingImageDataArray = NULL;

    if (!PyArg_ParseTuple(args, "OO", &staticImagePy, &movingImagePy)) return NULL;
    staticImageDataArray = transformToArray(staticImagePy);
    movingImageDataArray = transformToArray(movingImagePy);


    std::vector<int> dims;
    dims.push_back(PyArray_SHAPE(staticImageDataArray)[0]);
    dims.push_back(PyArray_SHAPE(staticImageDataArray)[1]);
    dims.push_back(PyArray_SHAPE(staticImageDataArray)[2]);

    Image<unsigned char> staticImage(dims);
    Image<unsigned char> movingImage(dims);

    loadImageData(staticImageDataArray, staticImage);
    loadImageData(movingImageDataArray, movingImage);

    Py_DECREF(staticImageDataArray);
    Py_DECREF(movingImageDataArray);
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

