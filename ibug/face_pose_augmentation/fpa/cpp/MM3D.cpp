#include <cmath>
#include <algorithm>
#include "MM3D.h"

using namespace std;

void MM3D::ZBuffer(const double *vertex, const int *tri, const double *texture, int nver, int ntri,
    const double *src_img, int width, int height, int nChannels, double *img, int *tri_ind)
{
    double *imgh = new double[width * height];
    for(int i = 0; i < width * height; ++i)
    {
        imgh[i] = -1.0e15;
        tri_ind[i] = -1;
    }
    if(src_img != NULL && img != NULL)
    {
        memcpy(img, src_img, width * height * nChannels * sizeof(double));
    }

    double coords[3] = {0.0, 0.0, 0.0};
    for(int i = 0; i < ntri; ++i)
    {
        const int &p1 = tri[i];
        const int &p2 = tri[i + ntri];
        const int &p3 = tri[i + ntri * 2];

        const double *pt1 = vertex + p1;
        const double *pt2 = vertex + p2;
        const double *pt3 = vertex + p3;

        const double *t1 = texture + nChannels * p1;
        const double *t2 = texture + nChannels * p2;
        const double *t3 = texture + nChannels * p3;

        int x_min = (int)ceil(min(min(pt1[0], pt2[0]), pt3[0]));
        int x_max = (int)floor(max(max(pt1[0], pt2[0]), pt3[0]));

        int y_min = (int)ceil(min(min(pt1[nver], pt2[nver]), pt3[nver]));
        int y_max = (int)floor(max(max(pt1[nver], pt2[nver]), pt3[nver]));

        x_min = min(max(x_min, 0), width - 1);
        x_max = min(max(x_max, 0), width - 1);
        y_min = min(max(y_min, 0), height - 1);
        y_max = min(max(y_max, 0), height - 1);

        for(int y = y_min; y <= y_max; ++y)
        {
            for(int x = x_min; x <= x_max; ++x)
            {
                ComputeBarycentricCoordinates(x, y, pt1, pt2, pt3, nver, coords);
                if(0.0 <= coords[0] && 0.0 <= coords[1] && 0.0 <= coords[2])
                {
                    double z = coords[0] * pt1[nver * 2] + coords[1] * pt2[nver * 2] + coords[2] * pt3[nver * 2];
                    if(imgh[y * width + x] < z)
                    {
                        imgh[y * width + x] = z;
                        if(texture != NULL && img != NULL)
                        {
                            for(int j = 0; j < nChannels; ++j)
                            {
                                img[(y * width + x) * nChannels + j] =
                                    coords[0] * t1[j] + coords[1] * t2[j] + coords[2] * t3[j];
                            }
                        }
                        tri_ind[y * width + x] = i;
                    }
                }
            }
        }
    }

    delete[] imgh;
}

void MM3D::ZBufferTri(const double *vertex, const int *tri, const double *texture_tri, int nver, int ntri,
    const double *src_img, int width, int height, int nChannels, double *img, int *tri_ind)
{
    double *imgh = new double[width * height];
    for(int i = 0; i < width * height; ++i)
    {
        imgh[i] = -1.0e15;
        tri_ind[i] = -1;
    }
    if(src_img != NULL && img != NULL)
    {
        memcpy(img, src_img, width * height * nChannels * sizeof(double));
    }

    double coords[3] = {0.0, 0.0, 0.0};
    for(int i = 0; i < ntri; ++i)
    {
        const double *pt1 = vertex + tri[i];
        const double *pt2 = vertex + tri[i + ntri];
        const double *pt3 = vertex + tri[i + ntri * 2];

        int x_min = (int)ceil(min(min(pt1[0], pt2[0]), pt3[0]));
        int x_max = (int)floor(max(max(pt1[0], pt2[0]), pt3[0]));

        int y_min = (int)ceil(min(min(pt1[nver], pt2[nver]), pt3[nver]));
        int y_max = (int)floor(max(max(pt1[nver], pt2[nver]), pt3[nver]));

        x_min = min(max(x_min, 0), width - 1);
        x_max = min(max(x_max, 0), width - 1);
        y_min = min(max(y_min, 0), height - 1);
        y_max = min(max(y_max, 0), height - 1);

        for(int y = y_min; y <= y_max; ++y)
        {
            for(int x = x_min; x <= x_max; ++x)
            {
                ComputeBarycentricCoordinates(x, y, pt1, pt2, pt3, nver, coords);
                if(0.0 <= coords[0] && 0.0 <= coords[1] && 0.0 <= coords[2])
                {
                    double z = coords[0] * pt1[nver * 2] + coords[1] * pt2[nver * 2] + coords[2] * pt3[nver * 2];
                    if(imgh[y * width + x] < z)
                    {
                        imgh[y * width + x] = z;
                        if(texture_tri != NULL && img != NULL)
                        {
                            for(int j = 0; j < nChannels; ++j)
                            {
                                img[(y * width + x) * nChannels + j] =  texture_tri[nChannels * i + j];
                            }
                        }
                        tri_ind[y * width + x] = i;
                    }
                }
            }
        }
    }

    delete[] imgh;
}

void MM3D::ComputeBarycentricCoordinates(
    double x, double y, const double *pt1, const double *pt2, const double *pt3, int nver, double coords[3])
{
    double det = (pt2[nver] - pt3[nver]) * (pt1[0] - pt3[0]) + (pt3[0] - pt2[0]) * (pt1[nver] - pt3[nver]);
    if(det == 0.0)
    {
        coords[0] = coords[1] = coords[2] = nan("");
    }
    else
    {
        coords[0] = ((pt2[nver] - pt3[nver]) * (x - pt3[0]) + (pt3[0] - pt2[0]) * (y - pt3[nver])) / det;
        coords[1] = ((pt3[nver] - pt1[nver]) * (x - pt3[0]) + (pt1[0] - pt3[0]) * (y - pt3[nver])) / det;
        coords[2] = 1.0 - coords[0] - coords[1];
    }
}
