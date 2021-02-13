#ifndef MM3D_H
#define MM3D_H

#include <math.h>
#include "MM3DGlobal.h"
// #include <opencv2/core/core.hpp>
// #include <opencv/cv.h>
#include <algorithm>
// #include <Windows.h>
// #include <gl\GL.h>
// #include <GL\GLU.h>

using namespace std;
// using namespace cv;

#define INF 1E20

class MM3D
{
public:
	// 3DMM and Reference Frame Mapping
	void Cartesian2Ref(double* vertex, double* tri, double* texture, int width, int height, int nver, int ntri, double* ref, double* refco, double* tri_ind);
	void ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);
	void ZBufferTri(double* vertex, double* tri, double* texture_tri, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);
	void GetCoverTri(double* vertex, double* tri, double* r, double* p, int nver, int ntri, double* coverTri);
	void VisibleSurf(double* vertex, double* tri, double* r, int nver, int ntri, double* vis_bin);
	// bool PointInTri(Mat* point, Mat* pt1, Mat* pt2, Mat* pt3);
	bool PointInTri(double point[], double pt1[], double pt2[], double pt3[]);
	void DistanceTransform(double* dt, double* im, int width, int height);
	void dt(float *d, float *f, int n); 

	// void Lighting(double* vertex, double* tri, double* tex, int _nv, int _nt, Illum_Para para, IplImage* img);
	void DrawModal(float* _shape, unsigned int* _triangle, float* _color, int _nt, int _nv, Illum_Para para, unsigned char* face, int widthstep);
	void NormDirection(float* vertex, unsigned int* tri, int nt, int nv, float* norm);

	void MeshMap(double* vertex, double* tri, int nver, int ntri, double* meshMap, int width, int height);
	
	void OcclusionQuery(double* vertex, double* tri, int ntri, int nver, int width, int height, double* visibility, double threshold);

// 	// OpenGL related
// 	HGLRC _hrc;	
// 	HDC _hdc;
// 	BYTE* _data;
// 	HBITMAP _bitmap;	// for face
// 	int _width;
// 	int _height;
// 	BOOL PrepareGL();
// 	BOOL ReleaseGL();

	// double xmin;
	// double xmax;
	// double ymin;
	// double ymax;
	// double zmin;
	// double zmax;
};



#endif