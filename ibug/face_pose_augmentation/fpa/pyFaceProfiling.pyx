import numpy as np
cimport numpy as np

cdef extern from "cpp/face_profiling.h":
    void TNorm2VNorm(double*, int, double*, double*, int)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# # IMPORTANT: as column-major is used in original mex functions, we need 
# #     ensure numpy array is converted to Fortran-styple order.

def pyTNorm2VNorm(np.ndarray[double, ndim=2, mode="fortran"] normt not None, 
                  np.ndarray[double, ndim=2, mode="fortran"] tri not None, 
                  ntri, nver):    
    cdef np.ndarray[double, ndim=2, mode="fortran"] normv    
    normv = np.zeros((3, nver), dtype=np.float64, order='F')

    # run C++
    TNorm2VNorm(&normv[0, 0], nver, &normt[0, 0], &tri[0, 0], ntri)

    return normv

