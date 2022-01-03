import numpy as np
cimport numpy as np


# declare FaceFrontalization class
cdef extern from "cpp/face_frontalization.h":
    cdef cppclass FaceFrontalization:
        FaceFrontalization()
        FaceFrontalization(const long *, int, int, const double *, const double *, int, const long *, int)
        FaceFrontalization(const double *, int, int, int, const double *)

        void frontalization_mapping(double *)
        void frontalization_filling(double *)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


def pyFaceFrontalizationMapping(np.ndarray[long, ndim=2, mode='c'] tri_ind not None, im_width, im_height,
                                np.ndarray[double, ndim=2, mode='c'] all_vertex_src not None,
                                np.ndarray[double, ndim=2, mode='c'] all_vertex_ref not None, all_ver_length,
                                np.ndarray[long, ndim=2, mode='c'] all_tri not None, all_tri_length):
    cdef np.ndarray[double, ndim=3, mode="c"] corres_map = np.zeros((im_height, im_width, 2), dtype=np.float64)

    cdef FaceFrontalization c_ff = FaceFrontalization(
        &tri_ind[0, 0], im_width, im_height,
        &all_vertex_src[0, 0], &all_vertex_ref[0, 0], all_ver_length,
        &all_tri[0, 0], all_tri_length)
    c_ff.frontalization_mapping(&corres_map[0, 0, 0])

    return corres_map


def pyFaceFrontalizationFilling(np.ndarray[double, ndim=3, mode='c'] img not None,
                                im_width, im_height, num_channels,
                                np.ndarray[double, ndim=3, mode='c'] corres_map not None):
    cdef np.ndarray[double, ndim=3, mode="c"] result = np.zeros((im_height, im_width, num_channels), dtype=np.float64)

    cdef FaceFrontalization c_ff = FaceFrontalization(
        &img[0, 0, 0], im_width, im_height, num_channels, &corres_map[0, 0, 0])
    c_ff.frontalization_filling(&result[0, 0, 0])

    return result
