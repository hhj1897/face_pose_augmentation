#ifndef MM3DGLOBAL_H
#define MM3DGLOBAL_H

#ifndef PI
#define PI 3.1415926
#endif

#ifndef MIN_PHI 
#define MIN_PHI -PI/2 - 20.0/180*PI
#endif

#ifndef MAX_PHI 
#define MAX_PHI PI/2 + 20.0/180*PI
#endif

#ifndef MIN_Y
#define MIN_Y -14 * 1e4 
#endif

#ifndef MAX_Y
#define MAX_Y 10 * 1e4;
#endif

#ifndef WIDTHBYTES
#define WIDTHBYTES(bits)    (((bits) + 31) >> 5 << 2)
#endif

typedef struct 
{
	float Amb_r;
	float Amb_g;
	float Amb_b;
	float Dir_r;
	float Dir_g;
	float Dir_b;
	float l[3];
}Illum_Para;

#endif