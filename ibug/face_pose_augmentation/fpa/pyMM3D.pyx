import numpy as np
cimport numpy as np

from libcpp cimport bool


# cdef extern from 'opencv2/core/core.hpp':
#     cdef int CV_64F

# cdef extern from 'opencv2/core/core.hpp' namespace "cv":
#     cdef cppclass Mat:
#         Mat() except +
#         void create(int, int, int)
#         void* data

#     cdef cppclass IplImage:
#         IplImage() except +

# declare 3DMM class
cdef extern from "cpp/MM3D.h":
    cdef cppclass MM3D:
        int _width, _height
        double xmin, xmax, ymin, ymax, zmin, zmax
        
        # 3DMM and Reference Frame Mapping
        void Cartesian2Ref(double*, double*, double*, int, int, int, int, double*, double*, double*)
        void ZBuffer(double*, double*, double*, int, int, double*, int, int, int, double*, double*)
        void ZBufferTri(double*, double*, double*, int, int, double*, int, int, int, double*, double*)
        void GetCoverTri(double*, double*, double*, double*, int, int, double*)
        void VisibleSurf(double*, double*, double*, int, int, double*)
        # bool PointInTri(Mat*, Mat*, Mat*, Mat*)
        bool PointInTri(double[], double[], double[], double[])
        void DistanceTransform(double*, double*, int, int)
        void dt(float*, float*, int)

        # void Lighting(double*, double*, double*, int, int, Illum_Para, IplImage*)
        void DrawModal(float*, unsigned int*, float*, int, int, Illum_Para, unsigned char*, int)
        void NormDirection(float*, unsigned int*, int, int, float*)

        void MeshMap(double*, double*, int, int, double*, int, int)
        
        void OcclusionQuery(double*, double*, int, int, int, int, double*, double)
        

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# # IMPORTANT: as column-major is used in original mex functions, we need 
# #     ensure numpy array is converted to Fortran-styple order.

cdef class pyMM3D:
    cdef MM3D c_3dmm

    def ZBuffer(self, np.ndarray[double, ndim=2, mode="fortran"] vertex not None, 
                  np.ndarray[double, ndim=2, mode="fortran"] tri not None, 
                  np.ndarray[double, ndim=2, mode="fortran"] texture not None, 
                  np.ndarray[double, ndim=3, mode="fortran"] src_img not None, 
                  nver, ntri, width, height, nChannels):
        cdef np.ndarray[double, ndim=3, mode="fortran"] img
        cdef np.ndarray[double, ndim=3, mode="fortran"] tri_ind

        img = np.zeros((height, width, nChannels), dtype=np.float64, order='F')
        tri_ind = np.zeros((height, width, 1), dtype=np.float64, order='F')

        self.c_3dmm.ZBuffer(&vertex[0,0], &tri[0,0], &texture[0,0], nver, ntri, 
                            &src_img[0,0,0], width, height, nChannels, 
                            &img[0,0,0], &tri_ind[0,0,0])

        return img, tri_ind

    def ZBufferTri(self, np.ndarray[double, ndim=2, mode="fortran"] vertex not None, 
                     np.ndarray[double, ndim=2, mode="fortran"] tri not None, 
                     np.ndarray[double, ndim=2, mode="fortran"] texture_tri not None, 
                     np.ndarray[double, ndim=3, mode="fortran"] src_img not None, 
                     nver, ntri, width, height, nChannels):
        cdef np.ndarray[double, ndim=3, mode="fortran"] img
        cdef np.ndarray[double, ndim=3, mode="fortran"] tri_ind

        img = np.zeros((height, width, nChannels), dtype=np.float64, order='F')
        tri_ind = np.zeros((height, width, 1), dtype=np.float64, order='F')

        self.c_3dmm.ZBufferTri(&vertex[0,0], &tri[0,0], &texture_tri[0,0], nver, ntri, 
                               &src_img[0,0,0], width, height, nChannels, 
                               &img[0,0,0], &tri_ind[0,0,0])

        return img, tri_ind