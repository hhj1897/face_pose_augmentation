import numpy as np
cimport numpy as np


# declare FaceFrontalization class
cdef extern from "cpp/face_frontalization.h":
    cdef cppclass FaceFrontalization:
        FaceFrontalization() except +
        
        FaceFrontalization(const double*, const double*, int, int, int, const double*, const double*, 
                            int, int, const double*, int, int, int, 
                            const double*, int, int, const double*, int) except +

        FaceFrontalization(const double*, const double*, int, int, int, const double*, const double*, 
                            int, int, const double*, int, int, int, 
                            const double*, int, int) except +

        FaceFrontalization(const double*, const double*, int, int, int, const double*, const double*, 
                            int, int, const double*, int, int, int, int, 
                            const double*, int, int, const double*, int) except +

        FaceFrontalization(const double*, int, int, int, double*) except +

        void frontalization_mapping(double*, double*)
        void frontalization_mapping_nosym(double*)
        void frontalization_mapping_big_tri(double*, double*)
        void frontalization_filling(double*)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# # IMPORTANT: as column-major is used in original mex functions, we need 
# #     ensure numpy array is converted to Fortran-styple order.

def pyFaceFrontalizationFilling(np.ndarray[double, ndim=3, mode='fortran'] img not None,
                                width, height, nChannels,
                                np.ndarray[double, ndim=3, mode='fortran'] corres_map not None):    
    cdef np.ndarray[double, ndim=3, mode="fortran"] result    
    result = np.zeros((height, width, nChannels), dtype=np.float64, order='F')

    cdef FaceFrontalization c_ff 
    c_ff = FaceFrontalization(&img[0, 0, 0], width, height, nChannels, &corres_map[0, 0, 0])

    c_ff.frontalization_filling(&result[0, 0, 0])

    return result


def pyFaceFrontalizationMappingBigTri(np.ndarray[double, ndim=2, mode='fortran'] mask not None,
                                      width, height, nChannels,
                                      np.ndarray[double, ndim=2, mode='fortran'] tri_ind not None,
                                      np.ndarray[double, ndim=2, mode='fortran'] all_vertex_src not None,
                                      np.ndarray[double, ndim=2, mode='fortran'] all_vertex_ref not None,
                                      all_ver_dim, all_ver_length,
                                      np.ndarray[double, ndim=2, mode='fortran'] all_tri not None,
                                      all_tri_dim, all_tri_length, bg_tri_num, bg_vertex_num,
                                      np.ndarray[double, ndim=2, mode='fortran'] valid_tri_half not None,
                                      vertex_length, tri_length,
                                      np.ndarray[double, ndim=2, mode='fortran'] sym_tri_list not None,
                                      symlist_length):    
    cdef np.ndarray[double, ndim=3, mode="fortran"] corres_map
    cdef np.ndarray[double, ndim=3, mode="fortran"] corres_map_sym

    corres_map = np.zeros((height, width, 2), dtype=np.float64, order='F')
    corres_map_sym = np.zeros((height, width, 2), dtype=np.float64, order='F')

    cdef FaceFrontalization c_ff 
    c_ff = FaceFrontalization(&mask[0, 0], &tri_ind[0, 0], width, height, nChannels, 
                              &all_vertex_src[0, 0], &all_vertex_ref[0, 0], all_ver_dim, all_ver_length,
                              &all_tri[0, 0], all_tri_dim, all_tri_length, bg_tri_num, bg_vertex_num,
                              &valid_tri_half[0, 0], vertex_length, tri_length,
                              &sym_tri_list[0, 0], symlist_length)

    c_ff.frontalization_mapping_big_tri(&corres_map[0, 0, 0], &corres_map_sym[0, 0, 0])

    return corres_map, corres_map_sym


def pyFaceFrontalizationMapping(np.ndarray[double, ndim=2, mode='fortran'] mask not None,
                                width, height, nChannels,
                                np.ndarray[double, ndim=2, mode='fortran'] tri_ind not None,
                                np.ndarray[double, ndim=2, mode='fortran'] all_vertex_src not None,
                                np.ndarray[double, ndim=2, mode='fortran'] all_vertex_ref not None,
                                all_ver_dim, all_ver_length,
                                np.ndarray[double, ndim=2, mode='fortran'] all_tri not None,
                                all_tri_dim, all_tri_length, bg_tri_num,
                                np.ndarray[double, ndim=2, mode='fortran'] valid_tri_half not None,
                                vertex_length, tri_length,
                                np.ndarray[double, ndim=2, mode='fortran'] sym_tri_list not None,
                                symlist_length):    
    cdef np.ndarray[double, ndim=3, mode="fortran"] corres_map
    cdef np.ndarray[double, ndim=3, mode="fortran"] corres_map_sym

    corres_map = np.zeros((height, width, 2), dtype=np.float64, order='F')
    corres_map_sym = np.zeros((height, width, 2), dtype=np.float64, order='F')

    cdef FaceFrontalization c_ff 
    c_ff = FaceFrontalization(&mask[0, 0], &tri_ind[0, 0], width, height, nChannels, 
                              &all_vertex_src[0, 0], &all_vertex_ref[0, 0], all_ver_dim, all_ver_length,
                              &all_tri[0, 0], all_tri_dim, all_tri_length, bg_tri_num,
                              &valid_tri_half[0, 0], vertex_length, tri_length,
                              &sym_tri_list[0, 0], symlist_length)

    c_ff.frontalization_mapping(&corres_map[0, 0, 0], &corres_map_sym[0, 0, 0])

    return corres_map, corres_map_sym    


def pyFaceFrontalizationMappingNosym(np.ndarray[double, ndim=2, mode='fortran'] mask not None,
                                    width, height, nChannels,
                                    np.ndarray[double, ndim=2, mode='fortran'] tri_ind not None,
                                    np.ndarray[double, ndim=2, mode='fortran'] all_vertex_src not None,
                                    np.ndarray[double, ndim=2, mode='fortran'] all_vertex_ref not None,
                                    all_ver_dim, all_ver_length,
                                    np.ndarray[double, ndim=2, mode='fortran'] all_tri not None,
                                    all_tri_dim, all_tri_length, bg_tri_num,
                                    np.ndarray[double, ndim=2, mode='fortran'] valid_tri_half not None,
                                    vertex_length, tri_length):    
    cdef np.ndarray[double, ndim=3, mode="fortran"] corres_map
    cdef np.ndarray[double, ndim=3, mode="fortran"] corres_map_sym

    corres_map = np.zeros((height, width, 2), dtype=np.float64, order='F')

    cdef FaceFrontalization c_ff 
    c_ff = FaceFrontalization(&mask[0, 0], &tri_ind[0, 0], width, height, nChannels, 
                              &all_vertex_src[0, 0], &all_vertex_ref[0, 0], all_ver_dim, all_ver_length,
                              &all_tri[0, 0], all_tri_dim, all_tri_length, bg_tri_num,
                              &valid_tri_half[0, 0], vertex_length, tri_length)

    c_ff.frontalization_mapping_nosym(&corres_map[0, 0, 0])

    return corres_map










