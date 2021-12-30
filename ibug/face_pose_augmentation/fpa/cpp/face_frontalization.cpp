#include <cmath>
#include "MM3D.h"
#include "face_frontalization.h"

using namespace std;

// Default constructor
FaceFrontalization::FaceFrontalization() {}

// Destructor
FaceFrontalization::~FaceFrontalization() {}

FaceFrontalization::FaceFrontalization(
    const long *tri_ind, int width, int height, const double *all_vertex_src,
    const double *all_vertex_ref, int all_ver_length, const long *all_tri, int all_tri_length)
{
    this->tri_ind = tri_ind;
    this->width = width;
    this->height = height;

    this->all_vertex_src = all_vertex_src;
    this->all_vertex_ref = all_vertex_ref;
    this->all_ver_length = all_ver_length;

    this->all_tri = all_tri;
    this->all_tri_length = all_tri_length;
}

FaceFrontalization::FaceFrontalization(
    const double *img, int width, int height, int nChannels, const double *corres_map)
{
    this->img = img;
    this->width = width;
    this->height = height;
    this->nChannels = nChannels;
    this->corres_map_input = corres_map;
}

void FaceFrontalization::frontalization_mapping(double *corres_map)
{
    double coords[3] = {0.0, 0.0, 0.0};
    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const long &corres_tri = tri_ind[y * width + x];
            if(corres_tri < 0)
            {
                corres_map[(y * width + x) * 2] = -1;
                corres_map[(y * width + x) * 2 + 1] = -1;
            }
            else
            {
                const double *pt1 = all_vertex_ref + all_tri[corres_tri];
                const double *pt2 = all_vertex_ref + all_tri[corres_tri + all_tri_length];
                const double *pt3 = all_vertex_ref + all_tri[corres_tri + all_tri_length * 2];
                MM3D::ComputeBarycentricCoordinates(x, y, pt1, pt2, pt3, all_ver_length, coords);

                pt1 = all_vertex_src + all_tri[corres_tri];
                pt2 = all_vertex_src + all_tri[corres_tri + all_tri_length];
                pt3 = all_vertex_src + all_tri[corres_tri + all_tri_length * 2];

                corres_map[(y * width + x) * 2] = coords[0] * pt1[0] + coords[1] * pt2[0] + coords[2] * pt3[0];
                corres_map[(y * width + x) * 2 + 1] = coords[0] * pt1[all_ver_length] +
                    coords[1] * pt2[all_ver_length] + coords[2] * pt3[all_ver_length];
            }
        }
    }
}

void FaceFrontalization::frontalization_filling(double *result)
{
    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const double &xx = corres_map_input[(y * width + x) * 2];
            const double &yy = corres_map_input[(y * width + x) * 2 + 1];
            if(xx < 0 || xx >= width || yy < 0 || yy >= height)
            {
                for(int n = 0; n < nChannels; ++n)
                {
                    result[(y * width + x) * nChannels + n] = 0;
                }
            }
            else
            {
                int left = (int)floor(xx);
                int right = (int)ceil(xx);
                int top = (int)floor(yy);
                int bottom = (int)ceil(yy);
                double delta_x = xx - left;
                double delta_y = yy - top;
                for(int n = 0; n < nChannels; ++n)
                {
                    const double &f00 = img[(top * width + left) * nChannels + n];
                    const double &f01 = img[(top * width + right) * nChannels + n];
                    const double &f10 = img[(bottom * width + left) * nChannels + n];
                    const double &f11 = img[(bottom * width + right) * nChannels + n];
                    result[(y * width + x) * nChannels + n] = f00 * (1.0 - delta_y) * (1.0 - delta_x) +
                        f01 * (1.0 - delta_y) * delta_x + f10 * delta_y * (1.0 - delta_x) + f11 * delta_y * delta_x;
                }
            }
        }
    }
}
