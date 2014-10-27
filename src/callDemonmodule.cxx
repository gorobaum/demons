#include <Python.h>
#include "calldemon.h"

static PyObject *
callDemon(PyObject *self, PyObject *args)
{
    int command;
    int sts;

    if (!PyArg_ParseTuple(args, "i", &command))
        return NULL;

    sts = callDemons(command);


    return Py_BuildValue("i", sts);
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
}

