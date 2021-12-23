#include <algorithm>
#include <math.h>
#include "face_frontalization.h"

using namespace std;


// Default constructor
FaceFrontalization::FaceFrontalization() {}

// Destructor
FaceFrontalization::~FaceFrontalization() {}

FaceFrontalization::FaceFrontalization(
    const long* tri_ind, int width, int height, const double* all_vertex_src, const double* all_vertex_ref,
    int all_ver_dim, int all_ver_length, const long* all_tri, int all_tri_dim, int all_tri_length)
{
    this->tri_ind = tri_ind;
    this->width = width;
    this->height = height;

    this->all_vertex_src = all_vertex_src;
    this->all_vertex_ref = all_vertex_ref;
    this->all_ver_dim = all_ver_dim;
    this->all_ver_length = all_ver_length;

    this->all_tri = all_tri;
    this->all_tri_dim = all_tri_dim;
    this->all_tri_length = all_tri_length;
}

FaceFrontalization::FaceFrontalization(const double* img, int width, int height, int nChannels, double* corres_map)
{
    this->img = img;
    this->width = width;
    this->height = height;
    this->nChannels = nChannels;
    this->corres_map_input = corres_map;
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
            corres_map[(y * width + x) * 2] = -1;
            corres_map[(y * width + x) * 2 + 1] = -1;

            corres_tri = tri_ind[y * width + x];

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

            corres_map[(y * width + x) * 2] = weight[0] * pt1[0] + weight[1] * pt2[0] + weight[2] * pt3[0];
            corres_map[(y * width + x) * 2 + 1] = weight[0] * pt1[1] + weight[1] * pt2[1] + weight[2] * pt3[1];
        }
    }
}

void FaceFrontalization::frontalization_filling(double* result)
{
    int x,y,n;
    double xx, yy;

    for(y = 0; y < height; y++)
    {
        for(x = 0; x < width; x++)
        {
            xx = corres_map_input[(y * width + x) * 2];
            yy = corres_map_input[(y * width + x) * 2 + 1];

            if(xx < 0 || x > width-1 || y < 0 || y > height-1)
            {
                for(n = 0; n < nChannels; n++){
                    result[(y * width + x) * nChannels + n] = 0;
                }
                continue;
            }

            bilinearInterpolation(result + (y * width + x) * nChannels, img, width, height, nChannels, xx, yy);
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

void FaceFrontalization::bilinearInterpolation(
    double* pixel, const double* img, int width, int height, int nChannels, double x, double y)
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
        f00[n] = img[((int)floor(y) * width + (int)floor(x)) * nChannels + n];
        f01[n] = img[((int)floor(y) * width + (int)ceil(x)) * nChannels + n];
        f10[n] = img[((int)ceil(y) * width + (int)floor(x)) * nChannels + n];
        f11[n] = img[((int)ceil(y) * width + (int)ceil(x)) * nChannels + n];
    }

    x = x - floor(x);
    y = y - floor(y);

    for(n = 0; n < nChannels; n++)
    {
        pixel[n] = f00[n] * (1 - y) * (1 - x) + f01[n] * (1 - y) * x + f10[n] * y * (1 - x) + f11[n] * y * x;
    }
}
