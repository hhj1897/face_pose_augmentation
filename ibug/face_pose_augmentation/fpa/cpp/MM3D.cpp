#include <algorithm>
#include "MM3D.h"

using namespace std;

void MM3D::ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri,
    double* src_img, int width, int height, int nChannels, double* img, double* tri_ind)
{
	double* imgh = new double[width * height];

	for(int i = 0; i < width * height; i++)
	{
		imgh[i] = -99999999999999;
		tri_ind[i] = -1;
	}

	//init image
	for(int i = 0; i < width * height * nChannels; i++)
	{
		img[i] = src_img[i];
	}

	for(int i = 0; i < ntri; i++)
	{
		int p1 = int(tri[3 * i]);
		int p2 = int(tri[3 * i + 1]);
		int p3 = int(tri[3 * i + 2]);

		double *pt1 = vertex + 3 * p1;
		double *pt2 = vertex + 3 * p2;
		double *pt3 = vertex + 3 * p3;

		double *t1 = texture + nChannels * p1;
		double *t2 = texture + nChannels * p2;
		double *t3 = texture + nChannels * p3;

		int x_min = (int)ceil(min(min(pt1[0], pt2[0]), pt3[0]));
		int x_max = (int)floor(max(max(pt1[0], pt2[0]), pt3[0]));

		int y_min = (int)ceil(min(min(pt1[1], pt2[1]), pt3[1]));
		int y_max = (int)floor(max(max(pt1[1], pt2[1]), pt3[1]));

		if(x_max < x_min || y_max < y_min)
			continue;

		x_min = min(max(x_min, 0), width-1);
		x_max = min(max(x_max, 0), width-1);
		y_min = min(max(y_min, 0), height-1);
		y_max = min(max(y_max, 0), height-1);
		
		for(int x = x_min; x <= x_max; x++)
		{
			for (int y = y_min; y <= y_max; y++)
			{
				double point[2] = {(double)x, (double)y};
				if (PointInTri(point, pt1, pt2, pt3))
				{
					double det = (pt2[1] - pt3[1]) * (pt1[0] - pt3[0]) + (pt3[0] - pt2[0]) * (pt1[1] - pt3[1]);
					double l1 = ((pt2[1] - pt3[1]) * (x - pt3[0]) + (pt3[0] - pt2[0]) * (y - pt3[1])) / det;
					double l2 = ((pt3[1] - pt1[1]) * (x - pt3[0]) + (pt1[0] - pt3[0]) * (y - pt3[1])) / det;
					double l3 = 1.0 - l1 - l2;
					double z = l1 * pt1[2] + l2 * pt2[2] + l3 * pt3[2];
					if( imgh[x * height + y] < z )
					{
						imgh[x * height + y] = z;
						for(int j = 0; j < nChannels; j++)
						{
							img[j * width * height + x * height + y] =  l1 * t1[j] + l2 * t2[j] + l3 * t3[j];
						}
						tri_ind[x * height + y] = i;
					}
				}
			}
		}
	}

	delete[] imgh;
}

void MM3D::ZBufferTri(double* vertex, double* tri, double* texture_tri, int nver, int ntri,
    double* src_img, int width, int height, int nChannels, double* img, double* tri_ind)
{
	double* imgh = new double[width * height];
	double* tritex = texture_tri;

	for(int i = 0; i < width * height; i++)
	{
		imgh[i] = -99999999999999;
		tri_ind[i] = -1;
	}

	//init image
	for(int i = 0; i < width * height * nChannels; i++)
	{
		img[i] = src_img[i];
	}

	for(int i = 0; i < ntri; i++)
	{
		double *pt1 = vertex + 3 * int(tri[3 * i]);
		double *pt2 = vertex + 3 * int(tri[3 * i + 1]);
		double *pt3 = vertex + 3 * int(tri[3 * i + 2]);

		int x_min = (int)ceil(min(min(pt1[0], pt2[0]), pt3[0]));
		int x_max = (int)floor(max(max(pt1[0], pt2[0]), pt3[0]));

		int y_min = (int)ceil(min(min(pt1[1], pt2[1]), pt3[1]));
		int y_max = (int)floor(max(max(pt1[1], pt2[1]), pt3[1]));

		if(x_max < x_min || y_max < y_min)
			continue;
		
		x_min = min(max(x_min, 0), width-1);
		x_max = min(max(x_max, 0), width-1);
		y_min = min(max(y_min, 0), height-1);
		y_max = min(max(y_max, 0), height-1);
		
		for(int x = x_min; x <= x_max; x++)
		{
			for (int y = y_min; y <= y_max; y++)
			{
				double point[2] = {(double)x, (double)y};
				if (PointInTri(point, pt1, pt2, pt3))
				{
					double det = (pt2[1] - pt3[1]) * (pt1[0] - pt3[0]) + (pt3[0] - pt2[0]) * (pt1[1] - pt3[1]);
					double l1 = ((pt2[1] - pt3[1]) * (x - pt3[0]) + (pt3[0] - pt2[0]) * (y - pt3[1])) / det;
					double l2 = ((pt3[1] - pt1[1]) * (x - pt3[0]) + (pt1[0] - pt3[0]) * (y - pt3[1])) / det;
					double l3 = 1.0 - l1 - l2;
					double z = l1 * pt1[2] + l2 * pt2[2] + l3 * pt3[2];
					if( imgh[x * height + y] < z )
					{
						imgh[x * height + y] = z;
						for(int j = 0; j < nChannels; j++)
						{
							img[j * width * height + x * height + y] =  tritex[nChannels * i + j];
						}
						tri_ind[x * height + y] = i;
					}
				}
			}
		}
	}

	delete[] imgh;
}

bool MM3D::PointInTri(double point[], double pt1[], double pt2[], double pt3[])
{
	double pointx = point[0];
	double pointy = point[1];

	double pt1x = pt1[0];
	double pt1y = pt1[1];

	double pt2x = pt2[0];
	double pt2y = pt2[1];

	double pt3x = pt3[0];
	double pt3y = pt3[1];

	double v0x = pt3x - pt1x;
	double v0y = pt3y - pt1y;

	double v1x = pt2x - pt1x;
	double v1y = pt2y - pt1y;

	double v2x = pointx - pt1x;
	double v2y = pointy - pt1y;

	double dot00 = v0x * v0x + v0y * v0y;
	double dot01 = v0x * v1x + v0y * v1y;
	double dot02 = v0x * v2x + v0y * v2y;
	double dot11 = v1x * v1x + v1y * v1y;
	double dot12 = v1x * v2x + v1y * v2y;

	double inverDeno = 0;
	if((dot00 * dot11 - dot01 * dot01) == 0)
		inverDeno = 0;
	else
		inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

	double u = (dot11 * dot02 - dot01 * dot12) * inverDeno;

	if(u < 0 || u > 1)
		return 0;

	double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

	if(v < 0 || v > 1)
		return 0;

	return u + v <= 1;
}
