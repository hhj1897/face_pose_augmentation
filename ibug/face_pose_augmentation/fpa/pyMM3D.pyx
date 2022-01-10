import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t


# declare 3DMM class
cdef extern from "cpp/MM3D.h":
    void _ZBuffer "MM3D::ZBuffer" (
        const double *, const int32_t *, const double *, int, int, const double *, int, int, int, double *, int32_t *)
    void _ZBufferTri "MM3D::ZBufferTri" (
        const double *, const int32_t *, const double *, int, int, const double *, int, int, int, double *, int32_t *)


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()


def ZBuffer(cnp.ndarray[double, ndim=2, mode="c"] vertex not None,
            cnp.ndarray[int32_t, ndim=2, mode="c"] tri not None,
            cnp.ndarray[double, ndim=2, mode="c"] texture,
            cnp.ndarray[double, ndim=3, mode="c"] src_img,
            nver, ntri, im_width, im_height, num_channels):
    cdef cnp.ndarray[double, ndim=3, mode="c"] img = None
    cdef cnp.ndarray[int32_t, ndim=2, mode="c"] tri_ind = np.zeros((im_height, im_width), dtype=np.int32_t)

    if texture is not None or src_img is not None:
        img = np.zeros((im_height, im_width, num_channels), dtype=np.float64)

    _ZBuffer(
        &vertex[0, 0], &tri[0, 0], <double *>(NULL) if texture is None else &texture[0, 0], nver, ntri,
        <double *>(NULL) if src_img is None else &src_img[0, 0, 0], im_width, im_height, num_channels,
        <double *>(NULL) if img is None else &img[0, 0, 0], &tri_ind[0, 0])

    return tri_ind, img


def ZBufferTri(cnp.ndarray[double, ndim=2, mode="c"] vertex not None,
               cnp.ndarray[int32_t, ndim=2, mode="c"] tri not None,
               cnp.ndarray[double, ndim=2, mode="c"] texture_tri,
               cnp.ndarray[double, ndim=3, mode="c"] src_img,
               nver, ntri, im_width, im_height, num_channels):
    cdef cnp.ndarray[double, ndim=3, mode="c"] img = None
    cdef cnp.ndarray[int32_t, ndim=2, mode="c"] tri_ind = np.zeros((im_height, im_width), dtype=np.int32_t)

    if texture_tri is not None and src_img is not None:
        img = np.zeros((im_height, im_width, num_channels), dtype=np.float64)

    _ZBufferTri(
        &vertex[0, 0], &tri[0, 0], <double *>(NULL) if texture_tri is None else &texture_tri[0, 0], nver, ntri,
        <double *>(NULL) if src_img is None else &src_img[0, 0, 0], im_width, im_height, num_channels,
        <double *>(NULL) if img is None else &img[0, 0, 0], &tri_ind[0, 0])

    return tri_ind, img
