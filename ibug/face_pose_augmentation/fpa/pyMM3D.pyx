import numpy as np
cimport numpy as cnp

# declare 3DMM class
cdef extern from "cpp/MM3D.h":
    void _ZBuffer "MM3D::ZBuffer" (double*, int*, double*, int, int, double*, int, int, int, double*, int*)
    void _ZBufferTri "MM3D::ZBufferTri" (double*, int*, double*, int, int, double*, int, int, int, double*, int*)

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

# # IMPORTANT: as column-major is used in original mex functions, we need 
# #     ensure numpy array is converted to Fortran-styple order.

def  ZBuffer(cnp.ndarray[double, ndim=2, mode="fortran"] vertex not None,
             cnp.ndarray[int, ndim=2, mode="fortran"] tri not None,
             cnp.ndarray[double, ndim=2, mode="fortran"] texture not None,
             cnp.ndarray[double, ndim=3, mode="fortran"] src_img not None,
             nver, ntri, width, height, nChannels):
    cdef cnp.ndarray[double, ndim=3, mode="fortran"] img
    cdef cnp.ndarray[int, ndim=3, mode="fortran"] tri_ind

    img = np.zeros((height, width, nChannels), dtype=np.float64, order='F')
    tri_ind = np.zeros((height, width, 1), dtype=np.int32, order='F')

    _ZBuffer(&vertex[0, 0], &tri[0, 0], &texture[0, 0], nver, ntri,
             &src_img[0, 0, 0], width, height, nChannels,
             &img[0, 0, 0], &tri_ind[0, 0, 0])

    return img, tri_ind

def ZBufferTri(cnp.ndarray[double, ndim=2, mode="fortran"] vertex not None,
               cnp.ndarray[int, ndim=2, mode="fortran"] tri not None,
               cnp.ndarray[double, ndim=2, mode="fortran"] texture_tri not None,
               cnp.ndarray[double, ndim=3, mode="fortran"] src_img not None,
               nver, ntri, width, height, nChannels):
    cdef cnp.ndarray[double, ndim=3, mode="fortran"] img
    cdef cnp.ndarray[int, ndim=3, mode="fortran"] tri_ind

    img = np.zeros((height, width, nChannels), dtype=np.float64, order='F')
    tri_ind = np.zeros((height, width, 1), dtype=np.int32, order='F')

    _ZBufferTri(&vertex[0, 0], &tri[0, 0], &texture_tri[0, 0], nver, ntri,
                &src_img[0, 0, 0], width, height, nChannels,
                &img[0, 0, 0], &tri_ind[0, 0, 0])

    return img, tri_ind
