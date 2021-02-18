#ifndef MM3D_H
#define MM3D_H

namespace MM3D {
	void ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri,
	    double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);
	void ZBufferTri(double* vertex, double* tri, double* texture_tri, int nver, int ntri,
	    double* src_img, int width, int height, int nChannels, double* img, double* tri_ind);
	bool PointInTri(double point[], double pt1[], double pt2[], double pt3[]);
};

#endif
