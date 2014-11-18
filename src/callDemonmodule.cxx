#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include "image.h"
#include "demonsfunction.h"
#include "symmetricdemonsfunction.h"
#include "asymmetricdemonsfunction.h"

static void
imageToNpArray(PyArrayObject * data, Image<unsigned char>& image) {
    std::vector<int> dims = image.getDimensions();
    for (int i = 0; i < dims[0]; i++)
        for (int j = 0; j < dims[1]; j++)
            for (int k = 0; k < dims[2]; k++) {
                *static_cast<unsigned char*>(PyArray_GETPTR3(data, i, j, k)) = image(i, j, k);
            }
}

static void
npArrayToImage(PyArrayObject * data, Image<unsigned char>& image) {
    std::vector<int> dims = image.getDimensions();
    for (int i = 0; i < dims[0]; i++)
        for (int j = 0; j < dims[1]; j++)
            for (int k = 0; k < dims[2]; k++) {
                unsigned char macaco = *static_cast<unsigned char*>(PyArray_GETPTR3(data, i, j, k));
                image(i, j, k) = macaco;
            }
}

PyArrayObject * transformToArray(PyObject * image) {
    PyObject * data = PyObject_CallMethod(image, "get_data", NULL);
    return (PyArrayObject*)PyArray_FROM_OTF(data, PyArray_DTYPE((PyArrayObject*)data)->type_num, 0);
}

Image<unsigned char>
applyVectorField(Image<unsigned char> image, VectorField displacementField) {
    std::vector<int> dimensions = image.getDimensions();
    Image<unsigned char> result(dimensions);
    Interpolation imageInterpolator(image);
    for(int x = 0; x < dimensions[0]; x++)
        for(int y = 0; y < dimensions[1]; y++)
            for(int z = 0; z < dimensions[2]; z++) {
                std::vector<double> displVector = displacementField.getVectorAt(x, y, z);
                double newX = x - displVector[0];
                double newY = y - displVector[1];
                double newZ = z - displVector[1];
                result(x,y,z) = imageInterpolator.trilinearInterpolation<unsigned char>(newX, newY, newZ);
            }
    return result;
}

extern "C"
PyObject *
callDemon(PyObject *self, PyObject *args) {
    PyObject * staticImagePy = NULL;
    PyObject * movingImagePy = NULL;
    PyArrayObject * aoutputArray = NULL;
    PyArrayObject * soutputArray = NULL;
    PyArrayObject * staticImageDataArray = NULL;
    PyArrayObject * movingImageDataArray = NULL;
    int iterations;
    double kernelSize, deviation, sx, sy, sz;
    std::vector<double> spacing;

    if (!PyArg_ParseTuple(args, "OOOOdddidd", &staticImagePy, &movingImagePy, &aoutputArray, &soutputArray, &sx, &sy, &sz, &iterations, &kernelSize, &deviation)) return NULL;
    std::cout << "iterations = " << iterations << "\n";
    std::cout << "kernelSize = " << kernelSize << "\n";
    std::cout << "deviation = " << deviation << "\n";

    spacing.push_back(sx);
    spacing.push_back(sy);
    spacing.push_back(sz);

    staticImageDataArray = transformToArray(staticImagePy);
    movingImageDataArray = transformToArray(movingImagePy);

    std::vector<int> dims;
    dims.push_back(PyArray_SHAPE(staticImageDataArray)[0]);
    dims.push_back(PyArray_SHAPE(staticImageDataArray)[1]);
    dims.push_back(PyArray_SHAPE(staticImageDataArray)[2]);

    Image<unsigned char> staticImage(dims);
    Image<unsigned char> movingImage(dims);


    npArrayToImage(staticImageDataArray, staticImage);
    npArrayToImage(movingImageDataArray, movingImage);

    // AsymmetricDemons aDemons(staticImage, movingImage, iterations, kernelSize, deviation, spacing);
    // aDemons.run();
    // VectorField aresultField = aDemons.getDisplField();
    // Image<unsigned char> aregistredImage = applyVectorField(movingImage, aresultField);
    // imageToNpArray(aoutputArray, aregistredImage);

    SymmetricDemonsFunction sDemons(staticImage, movingImage, spacing);
    sDemons.setExecutionParameters(iterations, 0);
    sDemons.setGaussianParameters(kernelSize, deviation);
    sDemons.run();
    VectorField sresultField = sDemons.getDisplField();
    Image<unsigned char> sregistredImage = applyVectorField(movingImage, sresultField);
    imageToNpArray(soutputArray, sregistredImage);

    // aresultField.printAround(122,57,102);
    // sresultField.printAround(122,57,102);
    // movingImage.printAround(122,57,102);
    // staticImage.printAround(122,57,102);
    // aregistredImage.printAround(122,57,102);
    // sregistredImage.printAround(122,57,102);


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

