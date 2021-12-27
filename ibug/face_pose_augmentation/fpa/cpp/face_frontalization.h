#ifndef FACE_FRONTALIZATION_H
#define FACE_FRONTALIZATION_H

#ifndef INF
#define INF 1E20
#endif

#ifndef MAXNCHANNELS
#define MAXNCHANNELS 5
#endif //MAXNCHANNELS

class FaceFrontalization
{
public:
    // Template
    FaceFrontalization();

    FaceFrontalization(
        const long* tri_ind, int width, int height, const double* all_vertex_src,
        const double* all_vertex_ref, int all_ver_length, const long* all_tri, int all_tri_length);

    FaceFrontalization(const double* img, int width, int height, int nChannels, double* corres_map);

    ~FaceFrontalization();

    void frontalization_mapping_nosym(double* corres_map);
    void frontalization_filling(double* result);

private:
    const double* img;
    const double* corres_map_input;
    int width;
    int height;
    int nChannels;

    // the map providing the corresponding tri of each pixel
    const long* tri_ind;

    // the meshed src and des vertices
    const double* all_vertex_src;
    const double* all_vertex_ref;
    int all_ver_dim;
    int all_ver_length;

    const long* all_tri;
    int all_tri_dim;
    int all_tri_length;

    // helper functions for frontalisation
    void position2weight(double* weight, double* point, double* pt1, double* pt2, double* pt3);
    void bilinearInterpolation(
        double* pixel, const double* img, int width, int height, int nChannels, double x, double y);
};

#endif
