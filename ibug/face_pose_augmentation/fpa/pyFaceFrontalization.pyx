import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t


# declare FaceFrontalization class
cdef extern from "cpp/face_frontalization.h":
    cdef cppclass FaceFrontalization:
        FaceFrontalization()
        FaceFrontalization(const int32_t *, int, int, const double *, const double *, int, const int32_t *, int)
        FaceFrontalization(const double *, int, int, int, const double *)

        void frontalization_mapping(double *)
        void frontalization_filling(double *)


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()


def pyFaceFrontalizationMapping(cnp.ndarray[int32_t, ndim=2, mode='c'] tri_ind not None, im_width, im_height,
                                cnp.ndarray[double, ndim=2, mode='c'] all_vertex_src not None,
                                cnp.ndarray[double, ndim=2, mode='c'] all_vertex_ref not None, all_ver_length,
                                cnp.ndarray[int32_t, ndim=2, mode='c'] all_tri not None, all_tri_length):
    cdef cnp.ndarray[double, ndim=3, mode="c"] corres_map = np.zeros((im_height, im_width, 2), dtype=np.float64)

    cdef FaceFrontalization c_ff = FaceFrontalization(
        &tri_ind[0, 0], im_width, im_height,
        &all_vertex_src[0, 0], &all_vertex_ref[0, 0], all_ver_length,
        &all_tri[0, 0], all_tri_length)
    c_ff.frontalization_mapping(&corres_map[0, 0, 0])

    return corres_map


def pyFaceFrontalizationFilling(cnp.ndarray[double, ndim=3, mode='c'] img not None, im_width, im_height, num_channels,
                                cnp.ndarray[double, ndim=3, mode='c'] corres_map not None):
    cdef cnp.ndarray[double, ndim=3, mode="c"] result = np.zeros((im_height, im_width, num_channels), dtype=np.float64)

    cdef FaceFrontalization c_ff = FaceFrontalization(
        &img[0, 0, 0], im_width, im_height, num_channels, &corres_map[0, 0, 0])
    c_ff.frontalization_filling(&result[0, 0, 0])

    return result
