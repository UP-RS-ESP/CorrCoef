#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#define VERSION "0.3"

PyArrayObject *
pearson(const double *d, const unsigned long n, const unsigned long l) {
	PyArrayObject *coef;
	double *c;
	npy_intp *dim;

	unsigned long ik, i, k, o, nn;
	double mk, sk, dk, h;
	double mi, si, sum;
	double *m, *s;

	nn = n * (n - 1) / 2;
	dim = malloc(sizeof(npy_intp));
	dim[0] = n * (n - 1) / 2;
	coef = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
	free(dim);
	if(!coef) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create output array.");
		return NULL;
	}

	/* mean and std */
	m = malloc(n * sizeof(double));
	s = malloc(n * sizeof(double));
	if(!m || !s) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create mean and std arrays.");
		return NULL;
	}
#pragma omp parallel for private(i, k, h, mk, sk, dk)
	for(i = 0; i < n; i++) {
		mk = sk = 0;
		for(k = 0; k < l; k++) {
			dk = d[i*l + k];
			h = dk - mk;
			mk += h / (k + 1);
			sk += h * (dk - mk);
		}
		m[i] = mk;
		s[i] = sqrt(sk / (l - 1));
	}

	/* dot products */
	c = (double *) coef->data;
#pragma omp parallel for private(ik, i, k, mi, si, mk, sk, o)
	for(ik = 0; ik < nn; ik++) {
		i = ik / n;
		k = ik % n;
		if(k <= i) {
			i = n - i - 2;
			k = n - k - 1;
		}
		mi = m[i];
		mk = m[k];
		si = s[i];
		sk = s[k];
		sum = 0;
#pragma omp parallel for reduction(+:sum)
		for(o = 0; o < l; o++)
			sum += (d[i*l + o] - mi) * (d[k*l + o] - mk) / si / sk;

		c[nn-(n-i)*((n-i)-1)/2+k-i-1] = sum / (l - 1);
	}
	free(m);
	free(s);

	return coef;
}

static PyObject *
CorrCoef_Pearson(PyObject *self, PyObject* args) {
	PyObject *arg;
	PyArrayObject *data, *coef;
	int nthreads;

	nthreads = 0;
	if(!PyArg_ParseTuple(args, "O|I", &arg, &nthreads))
		return NULL;
	data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
		PyArray_DOUBLE, 2, 2);
	if(!data)
		return NULL;
	if(nthreads)
		omp_set_num_threads(nthreads);
	
	coef = pearson((double *)data->data, data->dimensions[0], data->dimensions[1]);

	Py_DECREF(data);
	return PyArray_Return(coef);
}

static PyMethodDef CorrCoef_Methods[] = {
	{"Pearson", CorrCoef_Pearson, METH_VARARGS,
	 "triu_corr = Pearson(data, num_threads)\n\nReturn Pearson product-moment correlation coefficients.\n\nParameters\n----------\ndata : array_like\nA 2-D array containing multiple variables and observations. Each row of `data` represents a variable, and each column a single observation of all those variables.\n\nnum_threads : int, optional\nThe maximum number of OpenMP threads used.\n\nReturns\n-------\ntriu_corr : ndarray\nThe upper triangle of the correlation coefficient matrix of the variables.\n"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModDef = {
    PyModuleDef_HEAD_INIT,
    "CorrCoef",
    NULL,
    -1,
    CorrCoef_Methods
};

PyMODINIT_FUNC
PyInit_CorrCoef(void) {
    PyObject *mod;

    mod = PyModule_Create(&ModDef);
    PyModule_AddStringConstant(mod, "__author__", "Aljoscha Rheinwalt <aljoscha.rheinwalt@uni-potsdam.de>");
    PyModule_AddStringConstant(mod, "__version__", VERSION);
    import_array();

    return mod;
}

int
main(int argc, char **argv) {
    wchar_t pname[255];

    PyImport_AppendInittab("CorrCoef", PyInit_CorrCoef);
    mbstowcs(pname, argv[0], strlen(argv[0])+1);
    Py_SetProgramName(pname);
    Py_Initialize();
    PyImport_ImportModule("CorrCoef");
    PyMem_RawFree(argv[0]);
    return 0;
}
