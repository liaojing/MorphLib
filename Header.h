#ifndef CPUMORPH_PARAMETERS_H
#define CPUMORPH_PARAMETERS_H

#include <vector>
#include "time.h"
#include "Eigen\Dense"
using namespace Eigen;


enum BoundaryCondition
{
	BCOND_NONE,
	BCOND_CORNER,
	BCOND_BORDER
};

struct ConstraintPoint
{
	Vector2f lp, rp;
};

class CParameters
{
public:
	int h, w;//image size
	unsigned char *image0, *image1;
	unsigned char *mask0, *mask1;

	float w_ui, w_tps, w_ssim;
	float ssim_clamp;
	float eps;

	int max_iter;
	int start_res;
	float max_iter_drop_factor;

	BoundaryCondition bcond;

	std::vector<ConstraintPoint> ui_points;

	CParameters()
	{
		h = 0;
		w = 0;
		image0 = NULL;
		image1 = NULL;
		mask0 = NULL;
		mask1 = NULL;

		w_ui = 100.0f;
		w_tps = 0.001f;
		w_ssim = 1.0f;
		ssim_clamp = 0.0f;
		eps = 0.01f;
		start_res = 8;
		max_iter = 2000;
		max_iter_drop_factor = 2;
		bcond = BCOND_NONE;
		
	};

	~CParameters()
	{
	   if (image0)
		delete[] image0;
		if (image1)
		delete[] image1;
		if (mask0)
		delete[] mask0;
		if (mask1)
		delete[] mask1;
		image0 = NULL;
		image1 = NULL;
		mask0 = NULL;
		mask1 = NULL;
		ui_points.clear();
	};

};

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#endif