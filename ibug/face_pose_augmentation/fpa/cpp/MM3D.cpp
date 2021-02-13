#include "MM3D.h"


void MM3D::Cartesian2Ref(double* vertex, double* tri, double* texture, int width, int height, int nver, int ntri, double* ref, double* refco, double* tri_ind)
{
	int i,j;
	int x,y;

	double min_phi = MIN_PHI;
	double max_phi = MAX_PHI;

	double min_y = MIN_Y;
	double max_y = MAX_Y;

	double* u = new double[nver];
	double* v = new double[nver];

	double phi;


	for(i = 0; i < nver; i++)
	{
		phi = atan2(vertex[3*i], vertex[3*i+2]);
		u[i] = (phi - min_phi) / (max_phi - min_phi);
		u[i] = u[i] * (width - 1);
		v[i] = (vertex[3*i+1] - min_y) / (max_y - min_y);
		v[i] = v[i] * (height - 1);

		refco[2*i] = u[i];
		refco[2*i+1] = v[i];

	}

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* r = new double[ntri];
	double* refr = new double[width * height];
	double* tritex = new double[ntri * 3];

	for(i = 0; i < width * height; i++)
		refr[i] = 0;

	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i])-1;
		int p2 = int(tri[3*i+1])-1;
		int p3 = int(tri[3*i+2])-1;

		point1[2*i] = u[p1];	point1[2*i+1] = v[p1];
		point2[2*i] = u[p2];	point2[2*i+1] = v[p2];
		point3[2*i] = u[p3];	point3[2*i+1] = v[p3];

		double cent3d_x = (vertex[3*p1] + vertex[3*p2] + vertex[3*p3]) / 3;		
		double cent3d_z = (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

		r[i] = cent3d_x * cent3d_x + cent3d_z * cent3d_z;

		tritex[3*i] = (texture[3*p1] + texture[3*p2] + texture[3*p3]) / 3;
		tritex[3*i+1] = (texture[3*p1+1] + texture[3*p2+1] + texture[3*p3+1]) / 3;
		tritex[3*i+2] = (texture[3*p1+2] + texture[3*p2+2] + texture[3*p3+2]) / 3;
	}

	double point[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];

	for(i = 0; i < ntri; i++)
	{

		pt1[0] = point1[2*i]; pt1[1] = point1[2*i+1];
		pt2[0] = point2[2*i]; pt2[1] = point2[2*i+1];
		pt3[0] = point3[2*i]; pt3[1] = point3[2*i+1];
		
		int u_min = (int)ceil((double)min(min(pt1[0], pt2[0]), pt3[0]));
		int u_max = (int)floor((double)max(max(pt1[0], pt2[0]), pt3[0]));

		int v_min = (int)ceil((double)min(min(pt1[1], pt2[1]), pt3[1]));
		int v_max = (int)floor((double)max(max(pt1[1], pt2[1]), pt3[1]));

		if(u_max < u_min || v_max < v_min || u_max > width-1 || u_min < 0 || v_max > height-1 || v_min < 0)
			continue;
		
		for(x = u_min; x <= u_max; x++)
		{
			for (y = v_min; y <= v_max; y++)
			{
				point[0] = x;
				point[1] = y;
				if( refr[x * height + y] < r[i] && PointInTri(point, pt1, pt2, pt3))
				{
					refr[x * height + y] = r[i];
					for(j = 0; j < 3; j++)
					{
						ref[j * width * height + x * height + y] =  tritex[3 * i + j];
					}
					tri_ind[x * height + y] = i+1;
				}
			}
		}
	}

	delete[] u;
	delete[] v;
	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] r;
	delete[] refr;
	delete[] tritex;
}

// Get The Covering Triangle of point in p
// vertex: the coordinate of triangle's node
// tri: triangle's node index
// r: the 3d radius of triangle center
// coverTri: the index of covering triangle of point in p
void MM3D::GetCoverTri(double* vertex, double* tri, double* r, double* p, int nver, int ntri, double* coverTri)
{
	int i,j;

	//the coordinate of triangle's node
	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];

	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i])-1;
		int p2 = int(tri[3*i+1])-1;
		int p3 = int(tri[3*i+2])-1;

		point1[2*i] = vertex[2*p1]-1;	point1[2*i+1] = vertex[2*p1+1]-1;
		point2[2*i] = vertex[2*p2]-1;	point2[2*i+1] = vertex[2*p2+1]-1;
		point3[2*i] = vertex[2*p3]-1;	point3[2*i+1] = vertex[2*p3+1]-1;
	}

	double point[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];

	double* coverTriR = new double[nver];
	
	for(i = 0; i < nver; i++)
	{
		coverTriR[i] = 0;
		coverTri[i] = -1;
	}
	
	double x,y;

	for(i = 0; i < ntri; i++)
	{
		pt1[0] = point1[2*i]; pt1[1] = point1[2*i+1];
		pt2[0] = point2[2*i]; pt2[1] = point2[2*i+1];
		pt3[0] = point3[2*i]; pt3[1] = point3[2*i+1];
		
		// get the hull of triangle
		double x_min = (double)min(min(pt1[0], pt2[0]), pt3[0]);
		double x_max = (double)max(max(pt1[0], pt2[0]), pt3[0]);

		double y_min = (double)min(min(pt1[1], pt2[1]), pt3[1]);
		double y_max = (double)max(max(pt1[1], pt2[1]), pt3[1]);

		if(x_max < x_min || y_max < y_min )
			continue;

		for(j = 0; j < nver; j++)
		{
			// for every point in p
			x = p[2*j];
			y = p[2*j+1];
			if(x < x_min || x > x_max || y < y_min || y > y_max)
				continue;

			point[0] = x;
			point[1] = y;

			if(r[i] > coverTriR[j] && PointInTri(point, pt1, pt2, pt3))
			{
				coverTriR[j] = r[i];
				coverTri[j] = i;
			}

		}
		
	}

	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] coverTriR;
}

void MM3D::ZBuffer(double* vertex, double* tri, double* texture, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind)
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

void MM3D::ZBufferTri(double* vertex, double* tri, double* texture_tri, int nver, int ntri, double* src_img, int width, int height, int nChannels, double* img, double* tri_ind)
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

void MM3D::VisibleSurf(double* vertex, double* tri, double* triZ, int nver, int ntri, double* vis_bin)
{
	int i,j;

	//the coordinate of triangle's node
	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];

	double* pointh = new double[nver];

	for(i = 0; i < nver; i++)
	{
		pointh[i] = vertex[3*i+2]-1;
		vis_bin[i] = 1;
	}

	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i])-1;
		int p2 = int(tri[3*i+1])-1;
		int p3 = int(tri[3*i+2])-1;

		point1[2*i] = vertex[3*p1]-1;	point1[2*i+1] = vertex[3*p1+1]-1;
		point2[2*i] = vertex[3*p2]-1;	point2[2*i+1] = vertex[3*p2+1]-1;
		point3[2*i] = vertex[3*p3]-1;	point3[2*i+1] = vertex[3*p3+1]-1;
	}

	double point[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];

	
	double x,y;

	for(i = 5; i < ntri; i++)
	{
		pt1[0] = point1[2*i]; pt1[1] = point1[2*i+1];
		pt2[0] = point2[2*i]; pt2[1] = point2[2*i+1];
		pt3[0] = point3[2*i]; pt3[1] = point3[2*i+1];
		
		// get the hull of triangle
		double x_min = (double)min(min(point1[2*i], point2[2*i]), point3[2*i]);
		double x_max = (double)max(max(point1[2*i], point2[2*i]), point3[2*i]);

		double y_min = (double)min(min(point1[2*i+1], point2[2*i+1]), point3[2*i+1]);
		double y_max = (double)max(max(point1[2*i+1], point2[2*i+1]), point3[2*i+1]);

		int p1 = int(tri[3*i])-1;
		int p2 = int(tri[3*i+1])-1;
		int p3 = int(tri[3*i+2])-1;

		for(j = 3; j < nver; j++)
		{
			// if this point is vertex of the current triangle
			// or this point is invisible
			if(j == p1 || j == p2 || j == p3 || vis_bin[j] == 0)
				continue;

			x = vertex[3*j];
			y = vertex[3*j+1];

			if(x <= x_min || x >= x_max || y <= y_min || y >= y_max)
				continue;

			point[0] = x;
			point[1] = y;


			if(triZ[i] > pointh[j] && PointInTri(point, pt1, pt2, pt3))
			{
				// this point is covered by a triangle which is higher than the point
				vis_bin[j] = 0;
			}
		}
	}
	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] pointh;
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

void MM3D::MeshMap(double* vertex, double* tri, int nver, int ntri, double* meshMap, int width, int height)
{
	int i;
	int x,y;

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* h = new double[ntri];
	double* imgh = new double[width * height];

	for(i = 0; i < width * height; i++)
		imgh[i] = -99999999999999;

	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i])-1;
		int p2 = int(tri[3*i+1])-1;
		int p3 = int(tri[3*i+2])-1;

		point1[2*i] = vertex[3*p1]-1;	point1[2*i+1] = vertex[3*p1+1]-1;
		point2[2*i] = vertex[3*p2]-1;	point2[2*i+1] = vertex[3*p2+1]-1;
		point3[2*i] = vertex[3*p3]-1;	point3[2*i+1] = vertex[3*p3+1]-1;

		double cent3d_z = (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

		h[i] = cent3d_z;
	}

	double point[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];

	//init map
	for(i = 0; i < width * height; i++)
	{
		meshMap[i] = -1;
	}

	for(i = 0; i < ntri; i++)
	{
		pt1[0] = point1[2*i]; pt1[1] = point1[2*i+1];
		pt2[0] = point2[2*i]; pt2[1] = point2[2*i+1];
		pt3[0] = point3[2*i]; pt3[1] = point3[2*i+1];
		
		int x_min = (int)((double)min(min(pt1[0], pt2[0]), pt3[0]) + 0.5);
		int x_max = (int)((double)max(max(pt1[0], pt2[0]), pt3[0]) + 0.5);

		int y_min = (int)((double)min(min(pt1[1], pt2[1]), pt3[1]) + 0.5);
		int y_max = (int)((double)max(max(pt1[1], pt2[1]), pt3[1]) + 0.5);

		if(x_max < x_min || y_max < y_min || x_max > width-1 || x_min < 0 || y_max > height-1 || y_min < 0)
			continue;
		
		for(x = x_min; x <= x_max; x++)
		{
			for (y = y_min; y <= y_max; y++)
			{
				point[0] = x;
				point[1] = y;
				if( imgh[x * height + y] < h[i] && PointInTri(point, pt1, pt2, pt3))
				{
					imgh[x * height + y] = h[i];
					meshMap[x * height + y] = i+1;
				}
			}
		}
	}


	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] h;
	delete[] imgh;
}

void MM3D::OcclusionQuery(double* vertex, double* tri, int ntri, int nver, int width, int height, double* visibility, double threshold)
{
	int i;
	int x,y;

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* h = new double[ntri];
	double* imgh = new double[width * height];


	for(i = 0; i < width * height; i++)
		imgh[i] = -99999999999999;


	for(i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3*i])-1;
		int p2 = int(tri[3*i+1])-1;
		int p3 = int(tri[3*i+2])-1;

		point1[2*i] = vertex[3*p1]-1;	point1[2*i+1] = vertex[3*p1+1]-1;
		point2[2*i] = vertex[3*p2]-1;	point2[2*i+1] = vertex[3*p2+1]-1;
		point3[2*i] = vertex[3*p3]-1;	point3[2*i+1] = vertex[3*p3+1]-1;

		double cent3d_z = (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;

		h[i] = cent3d_z;
	}

	double point[2];
	double pt1[2];
	double pt2[2];
	double pt3[2];


	for(i = 0; i < ntri; i++)
	{
		pt1[0] = point1[2*i]; pt1[1] = point1[2*i+1];
		pt2[0] = point2[2*i]; pt2[1] = point2[2*i+1];
		pt3[0] = point3[2*i]; pt3[1] = point3[2*i+1];
		
		int x_min = (int)ceil((double)min(min(pt1[0], pt2[0]), pt3[0]));
		int x_max = (int)floor((double)max(max(pt1[0], pt2[0]), pt3[0]));

		int y_min = (int)ceil((double)min(min(pt1[1], pt2[1]), pt3[1]));
		int y_max = (int)floor((double)max(max(pt1[1], pt2[1]), pt3[1]));

		if(x_max < x_min || y_max < y_min || x_max > width-1 || x_min < 0 || y_max > height-1 || y_min < 0)
			continue;
		
		for(x = x_min; x <= x_max; x++)
		{
			for (y = y_min; y <= y_max; y++)
			{
				point[0] = x;
				point[1] = y;
				if( imgh[x * height + y] < h[i] && PointInTri(point, pt1, pt2, pt3))
				{
					imgh[x * height + y] = h[i];
				}
			}
		}
	}

	double vx,vy,vz;
	// get the visibility
	for(i = 0; i < nver; i++)
	{
		// get the coordinate
		vx = vertex[3*i]-1;
		vy = vertex[3*i+1]-1;
		vz = vertex[3*i+2]-1;

		// find the corres pixel
		int px = int(vx + 0.5);
		int py = int(vy + 0.5);

		if(abs(vz - imgh[px * height + py]) < threshold)
			visibility[i] = 1;
		else
			visibility[i] = 0;
	}

	delete[] h;
	delete[] imgh;
	delete[] point1;
	delete[] point2;
	delete[] point3;

}

