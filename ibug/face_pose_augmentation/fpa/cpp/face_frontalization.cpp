#include <algorithm>
#include <math.h>
#include "face_frontalization.h"

using namespace std;


// Default constructor
FaceFrontalization::FaceFrontalization() {}

// Destructor
FaceFrontalization::~FaceFrontalization() {}

FaceFrontalization::FaceFrontalization(const double* mask, const double* tri_ind, int width, int height, int nChannels, 
						const double* all_vertex_src, const double* all_vertex_ref, int all_ver_dim, int all_ver_length,
						const double* all_tri, int all_tri_dim, int all_tri_length,  int bg_tri_num,
						const double* valid_tri_half, int vertex_length, int tri_length,
						const double* sym_tri_list, int symlist_length)
{
	this->mask = mask;
	this->tri_ind = tri_ind;
	this->width = width;
	this->height = height;
	this->nChannels = nChannels;

	this->all_vertex_src = all_vertex_src;
	this->all_vertex_ref = all_vertex_ref;
	this->all_ver_dim = all_ver_dim;
	this->all_ver_length = all_ver_length;

	this->all_tri = all_tri;
	this->all_tri_dim = all_tri_dim;
	this->all_tri_length = all_tri_length;
	this->bg_tri_num = bg_tri_num;

	this->valid_tri_half = valid_tri_half;
	this->vertex_length = vertex_length;
	this->tri_length = tri_length;

	this->sym_tri_list = sym_tri_list;
	this->symlist_length = symlist_length;
}

FaceFrontalization::FaceFrontalization(const double* mask, const double* tri_ind, int width, int height, int nChannels, 
						const double* all_vertex_src, const double* all_vertex_ref, int all_ver_dim, int all_ver_length,
						const double* all_tri, int all_tri_dim, int all_tri_length, int bg_tri_num,
						const double* valid_tri_half, int vertex_length, int tri_length)
{
	this->mask = mask;
	this->tri_ind = tri_ind;
	this->width = width;
	this->height = height;
	this->nChannels = nChannels;

	this->all_vertex_src = all_vertex_src;
	this->all_vertex_ref = all_vertex_ref;
	this->all_ver_dim = all_ver_dim;
	this->all_ver_length = all_ver_length;

	this->all_tri = all_tri;
	this->all_tri_dim = all_tri_dim;
	this->all_tri_length = all_tri_length;
	this->bg_tri_num = bg_tri_num;

	this->valid_tri_half = valid_tri_half;
	this->vertex_length = vertex_length;
	this->tri_length = tri_length;
}

FaceFrontalization::FaceFrontalization(const double* mask, const double* tri_ind, int width, int height, int nChannels, 
						const double* all_vertex_src, const double* all_vertex_ref, int all_ver_dim, int all_ver_length,
						const double* all_tri, int all_tri_dim, int all_tri_length,  int bg_tri_num, int bg_vertex_num,
						const double* valid_tri_half, int vertex_length, int tri_length,
						const double* sym_list, int symlist_length)
{
	this->mask = mask;
	this->tri_ind = tri_ind;
	this->width = width;
	this->height = height;
	this->nChannels = nChannels;

	this->all_vertex_src = all_vertex_src;
	this->all_vertex_ref = all_vertex_ref;
	this->all_ver_dim = all_ver_dim;
	this->all_ver_length = all_ver_length;

	this->all_tri = all_tri;
	this->all_tri_dim = all_tri_dim;
	this->all_tri_length = all_tri_length;
	this->bg_tri_num = bg_tri_num;
	this->bg_vertex_num = bg_vertex_num;

	this->valid_tri_half = valid_tri_half;
	this->vertex_length = vertex_length;
	this->tri_length = tri_length;

	this->sym_list = sym_list;
	this->symlist_length = symlist_length;
}

FaceFrontalization::FaceFrontalization(const double* img, int width, int height, int nChannels, double* corres_map)
{
	this->img = img;
	this->width = width;
	this->height = height;
	this->nChannels = nChannels;
	this->corres_map_input = corres_map;
}

void FaceFrontalization::frontalization_mapping(double* corres_map, double* corres_map_sym)
{
	int x,y;
	double xx, yy;
	int corres_tri;
	int sym_tri;

	double weight[3];

	int pt_ind;

	double pt[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];

	
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			corres_map[0 * width * height + x * height + y] = -1;
			corres_map[1 * width * height + x * height + y] = -1;
			
			corres_map_sym[0 * width * height + x * height + y] = -1;
			corres_map_sym[1 * width * height + x * height + y] = -1;


			corres_tri = (int)tri_ind[x * height + y];

			if(corres_tri < 0)
			{
				// for no tri corresponding pixel
				continue;
			}

// 			//1. For des_img
// 			if(corres_tri < bg_tri_num)
// 			{
// 				// if it is on the background (big tri)
// 				// Positions on the des image
// 				pt_ind = (int)all_tri[all_tri_dim * corres_tri + 0];
// 				pt1[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
// 				pt1[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];
// 
// 				pt_ind = (int)all_tri[all_tri_dim * corres_tri + 1];
// 				pt2[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
// 				pt2[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];
// 
// 				pt_ind = (int)all_tri[all_tri_dim * corres_tri + 2];
// 				pt3[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
// 				pt3[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];
// 
// 				pt[0] = x;
// 				pt[1] = y;
// 				FaceFrontalization::position2weight(weight, pt, pt1, pt2, pt3);
// 			}
// 			else
// 			{
// 				// if it is on the face (small tri)
// 				weight[0] = weight[1] = weight[2] = 1.0 / 3; 
// 			}
            // I think this the correct ones - Shiyang
            // if it is on the background (big tri); Positions on the des image
            pt_ind = (int)all_tri[all_tri_dim * corres_tri + 0];
            pt1[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
            pt1[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];

            pt_ind = (int)all_tri[all_tri_dim * corres_tri + 1];
            pt2[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
            pt2[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];

            pt_ind = (int)all_tri[all_tri_dim * corres_tri + 2];
            pt3[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
            pt3[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];

            pt[0] = x;
            pt[1] = y;
            FaceFrontalization::position2weight(weight, pt, pt1, pt2, pt3);

			// Positions on the src img
			pt_ind = (int)all_tri[all_tri_dim * corres_tri + 0];
			pt1[0] = all_vertex_src[all_ver_dim * pt_ind + 0];
			pt1[1] = all_vertex_src[all_ver_dim * pt_ind + 1];

			pt_ind = (int)all_tri[all_tri_dim * corres_tri + 1];
			pt2[0] = all_vertex_src[all_ver_dim * pt_ind + 0];
			pt2[1] = all_vertex_src[all_ver_dim * pt_ind + 1];

			pt_ind = (int)all_tri[all_tri_dim * corres_tri + 2];
			pt3[0] = all_vertex_src[all_ver_dim * pt_ind + 0];
			pt3[1] = all_vertex_src[all_ver_dim * pt_ind + 1];

			xx = weight[0] * pt1[0] + weight[1] * pt2[0] + weight[2] * pt3[0];
			yy = weight[0] * pt1[1] + weight[1] * pt2[1] + weight[2] * pt3[1];

			corres_map_sym[0 * width * height + x * height + y] = xx;
			corres_map_sym[1 * width * height + x * height + y] = yy;

			if(mask[x * height + y] <= 0)
			{
				corres_map[0 * width * height + x * height + y] = xx;
				corres_map[1 * width * height + x * height + y] = yy;
			}


			// For sym_img
			if(corres_tri >= bg_tri_num && valid_tri_half[corres_tri - bg_tri_num] <= 0)
			{
				sym_tri = (int)FindSym(sym_tri_list, symlist_length, corres_tri-bg_tri_num) + bg_tri_num;
				xx = 0; yy = 0;

				pt_ind = (int)all_tri[all_tri_dim * sym_tri + 0];
				xx += all_vertex_src[all_ver_dim * pt_ind + 0];
				yy += all_vertex_src[all_ver_dim * pt_ind + 1];

				pt_ind = (int)all_tri[all_tri_dim * sym_tri + 1];
				xx += all_vertex_src[all_ver_dim * pt_ind + 0];
				yy += all_vertex_src[all_ver_dim * pt_ind + 1];

				pt_ind = (int)all_tri[all_tri_dim * sym_tri + 2];
				xx += all_vertex_src[all_ver_dim * pt_ind + 0];
				yy += all_vertex_src[all_ver_dim * pt_ind + 1];

				xx = xx / 3;
				yy = yy / 3;

				corres_map_sym[0 * width * height + x * height + y] = xx;
				corres_map_sym[1 * width * height + x * height + y] = yy;

			}
		}
	}
}

void FaceFrontalization::frontalization_mapping_nosym(double* corres_map)
{
	int x,y;
	double xx, yy;
	int corres_tri;

	double weight[3];

	int pt_ind;

	double pt[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];
	
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			corres_map[0 * width * height + x * height + y] = -1;
			corres_map[1 * width * height + x * height + y] = -1;

			corres_tri = (int)tri_ind[x * height + y];

			if(corres_tri < 0)
			{
				// for no tri corresponding pixel
				continue;
			}

            // I think this the correct ones - Shiyang
            // if it is on the background (big tri); Positions on the des image
            pt_ind = (int)all_tri[all_tri_dim * corres_tri + 0];
            pt1[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
            pt1[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];

            pt_ind = (int)all_tri[all_tri_dim * corres_tri + 1];
            pt2[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
            pt2[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];

            pt_ind = (int)all_tri[all_tri_dim * corres_tri + 2];
            pt3[0] = all_vertex_ref[all_ver_dim * pt_ind + 0];
            pt3[1] = all_vertex_ref[all_ver_dim * pt_ind + 1];

            pt[0] = x;
            pt[1] = y;
            FaceFrontalization::position2weight(weight, pt, pt1, pt2, pt3);

			// Positions on the src img
			pt_ind = (int)all_tri[all_tri_dim * corres_tri + 0];
			pt1[0] = all_vertex_src[all_ver_dim * pt_ind + 0];
			pt1[1] = all_vertex_src[all_ver_dim * pt_ind + 1];

			pt_ind = (int)all_tri[all_tri_dim * corres_tri + 1];
			pt2[0] = all_vertex_src[all_ver_dim * pt_ind + 0];
			pt2[1] = all_vertex_src[all_ver_dim * pt_ind + 1];

			pt_ind = (int)all_tri[all_tri_dim * corres_tri + 2];
			pt3[0] = all_vertex_src[all_ver_dim * pt_ind + 0];
			pt3[1] = all_vertex_src[all_ver_dim * pt_ind + 1];

			xx = weight[0] * pt1[0] + weight[1] * pt2[0] + weight[2] * pt3[0];
			yy = weight[0] * pt1[1] + weight[1] * pt2[1] + weight[2] * pt3[1];

			if(mask[x * height + y] <= 0)
			{
				corres_map[0 * width * height + x * height + y] = xx;
				corres_map[1 * width * height + x * height + y] = yy;
			}
		}
	}
}

void FaceFrontalization::frontalization_mapping_big_tri(double* corres_map, double* corres_map_sym)
{
	int x,y;
	double xx, yy;
	int corres_tri;

	double weight[3];

	int pt_ind[3];
	int ppt_ind;

	double pt[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];

	
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			corres_map[0 * width * height + x * height + y] = -1;
			corres_map[1 * width * height + x * height + y] = -1;
			
			corres_map_sym[0 * width * height + x * height + y] = -1;
			corres_map_sym[1 * width * height + x * height + y] = -1;


			corres_tri = (int)tri_ind[x * height + y];

			if(corres_tri < 0)
			{
				// for no tri corresponding pixel
				continue;
			}

			//1. For des_img
			pt_ind[0] = (int)all_tri[all_tri_dim * corres_tri + 0];
			pt1[0] = all_vertex_ref[all_ver_dim * pt_ind[0] + 0];
			pt1[1] = all_vertex_ref[all_ver_dim * pt_ind[0] + 1];

			pt_ind[1] = (int)all_tri[all_tri_dim * corres_tri + 1];
			pt2[0] = all_vertex_ref[all_ver_dim * pt_ind[1] + 0];
			pt2[1] = all_vertex_ref[all_ver_dim * pt_ind[1] + 1];

			pt_ind[2] = (int)all_tri[all_tri_dim * corres_tri + 2];
			pt3[0] = all_vertex_ref[all_ver_dim * pt_ind[2] + 0];
			pt3[1] = all_vertex_ref[all_ver_dim * pt_ind[2] + 1];

			pt[0] = x;
			pt[1] = y;
			FaceFrontalization::position2weight(weight, pt, pt1, pt2, pt3);


			// Positions on the src img
			pt1[0] = all_vertex_src[all_ver_dim * pt_ind[0] + 0];
			pt1[1] = all_vertex_src[all_ver_dim * pt_ind[0] + 1];

			pt2[0] = all_vertex_src[all_ver_dim * pt_ind[1] + 0];
			pt2[1] = all_vertex_src[all_ver_dim * pt_ind[1] + 1];

			pt3[0] = all_vertex_src[all_ver_dim * pt_ind[2] + 0];
			pt3[1] = all_vertex_src[all_ver_dim * pt_ind[2] + 1];

			xx = weight[0] * pt1[0] + weight[1] * pt2[0] + weight[2] * pt3[0];
			yy = weight[0] * pt1[1] + weight[1] * pt2[1] + weight[2] * pt3[1];

			corres_map_sym[0 * width * height + x * height + y] = xx;
			corres_map_sym[1 * width * height + x * height + y] = yy;

			if(mask[x * height + y] <= 0)
			{
				corres_map[0 * width * height + x * height + y] = xx;
				corres_map[1 * width * height + x * height + y] = yy;
			}


			// For sym_img
			if(corres_tri >= bg_tri_num && valid_tri_half[corres_tri - bg_tri_num] <= 0)
			{
				
				xx = 0; yy = 0;

				ppt_ind = (int)FindSym(sym_list, symlist_length, pt_ind[0]-bg_vertex_num) + bg_vertex_num;
                pt1[0] = all_vertex_src[all_ver_dim * ppt_ind + 0];
                pt1[1] = all_vertex_src[all_ver_dim * ppt_ind + 1];

				ppt_ind = (int)FindSym(sym_list, symlist_length, pt_ind[1]-bg_vertex_num) + bg_vertex_num;
				pt2[0] = all_vertex_src[all_ver_dim * ppt_ind + 0];
                pt2[1] = all_vertex_src[all_ver_dim * ppt_ind + 1];

				ppt_ind = (int)FindSym(sym_list, symlist_length, pt_ind[2]-bg_vertex_num) + bg_vertex_num;
				pt3[0] = all_vertex_src[all_ver_dim * ppt_ind + 0];
                pt3[1] = all_vertex_src[all_ver_dim * ppt_ind + 1];

				xx = weight[0] * pt1[0] + weight[1] * pt2[0] + weight[2] * pt3[0];
                yy = weight[0] * pt1[1] + weight[1] * pt2[1] + weight[2] * pt3[1];

				corres_map_sym[0 * width * height + x * height + y] = xx;
				corres_map_sym[1 * width * height + x * height + y] = yy;

			}
		}
	}
}

void FaceFrontalization::frontalization_filling(double* result)
{
	int x,y,n;
	double xx, yy;

	double pixel[MAXNCHANNELS];
	
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			xx = corres_map_input[0 * width * height + x * height + y];
			yy = corres_map_input[1 * width * height + x * height + y];

			if(xx < 0 || x > width-1 || y < 0 || y > height-1)
			{
				for(n = 0; n < nChannels; n++){
					result[n * width * height + x * height + y] = 0;
				}
				continue;
			}

			bilinearInterpolation(pixel, img, width, height, nChannels, xx, yy);

			for(n = 0; n < nChannels; n++)
				result[n * width * height + x * height + y] = pixel[n];
		}
	}
}


void FaceFrontalization::position2weight(double* weight, double* point, double* pt1, double* pt2, double* pt3)
{
	//Use the center of gravity to get the weights of three vertices of triangle
    //Ref: http://www.nowamagic.net/algorithm/algorithm_PointInTriangleTest.php
 
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
	double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

	weight[0] = 1 - u - v;
	weight[1] = v;
	weight[2] = u;
}

void FaceFrontalization::bilinearInterpolation(double* pixel, const double* img, int width, int height, int nChannels, double x, double y)
{
	int n;
	double f00[MAXNCHANNELS], f01[MAXNCHANNELS], f10[MAXNCHANNELS], f11[MAXNCHANNELS];
	/*x = min(max(0.0,x),width-1.0); 
	y = min(max(0.0,y),height-1.0); */

	if(x < 0 || x > width-1 || y < 0 || y > height-1)
	{
		for(n = 0; n < nChannels; n++)
		{	
			pixel[n] = 0;
		}
		return;
	}

	for(n = 0; n < nChannels; n++)
	{
		f00[n] = img[n * height * width + (int)floor(x) * height + (int)floor(y)];
		f01[n] = img[n * height * width + (int)floor(x) * height + (int)ceil(y)];
		f10[n] = img[n * height * width + (int)ceil(x) * height + (int)floor(y)];
		f11[n] = img[n * height * width + (int)ceil(x) * height + (int)ceil(y)];
	}

	x = x - floor(x);
	y = y - floor(y);

	for(n = 0; n < nChannels; n++)
	{
		pixel[n] = f00[n]*(1-x)*(1-y) + f01[n]*(1-x)*y + f10[n]*x*(1-y) + f11[n]*x*y;
	}

}

double FaceFrontalization::FindSym(const double* sym_list, int listlength, int ind)
{
	int i;
	for(i = 0; i < listlength; i++)
	{
		if(fabs(sym_list[2*i+0] - ind) < 1e-5)
		{
			return sym_list[2*i+1];
		}

		if(fabs(sym_list[2*i+1] - ind) < 1e-5)
		{
			return sym_list[2*i+0];
		}
	}

	return INF;
}
