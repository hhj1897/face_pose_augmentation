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

	FaceFrontalization(const double* mask, const double* tri_ind, int width, int height, int nChannels, 
						const double* all_vertex_src, const double* all_vertex_ref, int all_ver_dim, int all_ver_length,
						const double* all_tri, int all_tri_dim, int all_tri_length,  int bg_tri_num,
						const double* valid_tri_half, int vertex_length, int tri_length,
						const double* sym_tri_list, int symlist_length);

	FaceFrontalization(const double* mask, const double* tri_ind, int width, int height, int nChannels,
						const double* all_vertex_src, const double* all_vertex_ref, int all_ver_dim, int all_ver_length,
						const double* all_tri, int all_tri_dim, int all_tri_length,  int bg_tri_num,
						const double* valid_tri_half, int vertex_length, int tri_length);										

	FaceFrontalization(const double* mask, const double* tri_ind, int width, int height, int nChannels,
						const double* all_vertex_src, const double* all_vertex_ref, int all_ver_dim, int all_ver_length,
						const double* all_tri, int all_tri_dim, int all_tri_length,  int bg_tri_num, int bg_vertex_num,
						const double* valid_tri_half, int vertex_length, int tri_length,
						const double* sym_list, int symlist_length);			

	FaceFrontalization(const double* img, int width, int height, int nChannels, double* corres_map);

	~FaceFrontalization();

	// Part 1. Frontalization
	void frontalization_mapping(double* corres_map, double* corre_map_sym);
	void frontalization_mapping_nosym(double* corres_map);
	void frontalization_mapping_big_tri(double* corres_map, double* corre_map_sym);
	void frontalization_filling(double* result);

private:
	const double* img;
	const double* mask;
	const double* corres_map_input;
	int width;
	int height;
	int nChannels;

	///// For meshed image
	// the map providing the corresponding tri of each pixel
	const double* tri_ind;

	// the meshed src and des image
	const double* all_vertex_src;
	const double* all_vertex_ref;
	int all_ver_dim;
	int all_ver_length;

	const double* all_tri;
	int all_tri_dim;
	int all_tri_length;

	int bg_tri_num; // the number of background tri
	int bg_vertex_num;

	//// For face symmetry
	const double* valid_tri_half; // the visible half (set to 1) and invisible half (set to 0)
	int vertex_length;
	int tri_length;

	const double* sym_tri_list;
	const double* sym_list;
	int symlist_length;

	// Part 1. Frontalization
	void position2weight(double* weight, double* point, double* pt1, double* pt2, double* pt3);
	void bilinearInterpolation(double* pixel, const double* img, int width, int height, int nChannels, double x, double y);
	double FindSym(const double* sym_list, int listlength, int ind);
};

#endif