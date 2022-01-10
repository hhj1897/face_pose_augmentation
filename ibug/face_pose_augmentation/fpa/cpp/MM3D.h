#ifndef MM3D_H
#define MM3D_H

#include <cstdint>

namespace MM3D {
    void ZBuffer(const double *vertex, const std::int32_t *tri, const double *texture, int nver, int ntri,
        const double *src_img, int width, int height, int nChannels, double *img, std::int32_t *tri_ind);
    void ZBufferTri(const double *vertex, const std::int32_t *tri, const double *texture_tri, int nver, int ntri,
        const double *src_img, int width, int height, int nChannels, double *img, std::int32_t *tri_ind);
    void ComputeBarycentricCoordinates(
        double x, double y, const double *pt1, const double *pt2, const double *pt3, int nver, double coords[3]);
};

#endif
