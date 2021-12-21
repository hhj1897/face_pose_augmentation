import numpy as np
cimport numpy as cnp

# declare 3DMM class
cdef extern from "cpp/MM3D.h":
    void _ZBuffer "MM3D::ZBuffer" (double*, long*, double*, int, int, double*, int, int, int, double*, long*)
    void _ZBufferTri "MM3D::ZBufferTri" (double*, long*, double*, int, int, double*, int, int, int, double*, long*)

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

def ZBuffer(cnp.ndarray[double, ndim=2, mode="c"] vertex not None,
            cnp.ndarray[long, ndim=2, mode="c"] tri not None,
            cnp.ndarray[double, ndim=2, mode="c"] texture not None,
            cnp.ndarray[double, ndim=3, mode="c"] src_img not None,
            nver, ntri, width, height, nChannels):
    cdef cnp.ndarray[double, ndim=3, mode="c"] img
    cdef cnp.ndarray[long, ndim=3, mode="c"] tri_ind

    img = np.zeros((height, width, nChannels), dtype=np.float64)
    tri_ind = np.zeros((height, width, 1), dtype=np.int32)

    _ZBuffer(&vertex[0, 0], &tri[0, 0], &texture[0, 0], nver, ntri,
             &src_img[0, 0, 0], width, height, nChannels,
             &img[0, 0, 0], &tri_ind[0, 0, 0])

    return img, tri_ind

def ZBufferTri(cnp.ndarray[double, ndim=2, mode="c"] vertex not None,
               cnp.ndarray[long, ndim=2, mode="c"] tri not None,
               cnp.ndarray[double, ndim=2, mode="c"] texture_tri not None,
               cnp.ndarray[double, ndim=3, mode="c"] src_img not None,
               nver, ntri, width, height, nChannels):
    cdef cnp.ndarray[double, ndim=3, mode="c"] img
    cdef cnp.ndarray[long, ndim=3, mode="c"] tri_ind

    img = np.zeros((height, width, nChannels), dtype=np.float64)
    tri_ind = np.zeros((height, width, 1), dtype=np.int32)

    _ZBufferTri(&vertex[0, 0], &tri[0, 0], &texture_tri[0, 0], nver, ntri,
                &src_img[0, 0, 0], width, height, nChannels,
                &img[0, 0, 0], &tri_ind[0, 0, 0])

    return img, tri_ind
