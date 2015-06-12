#include "Header.h"
#include "Pyramid.h"
#include "..\Pyramids\extension.h"
#include "..\Pyramids\kernel.h"
#include "..\Pyramids\discrete.h"
#include "..\Pyramids\error.h"
#include "..\Pyramids\scale.h"
#include <iostream>

CPyramid::CPyramid()
{
}


CPyramid::~CPyramid()
{
	for (int i = 0; i < levels.size(); i++)
	{
		if (levels[i].image0)
			delete[] levels[i].image0;
		if (levels[i].image1)
			delete[] levels[i].image1;
		if (levels[i].mask0)
			delete[] levels[i].mask0;
		if (levels[i].mask1)
			delete[] levels[i].mask1;

		levels[i].image0 = NULL;
		levels[i].image1 = NULL;
		levels[i].mask0 = NULL;
		levels[i].mask1 = NULL;

		
	}
	levels.clear();
}


void CPyramid::create_pyramid(int w, int h, unsigned char* image0, unsigned char* image1, unsigned char* mask0, unsigned char* mask1, int start_res)
{
	std::cout << "Building pyramids \n";

		// use cardinal bspline3 prefilter for downsampling
	kernel::base *pre = new kernel::generalized(
		new kernel::discrete::delta,
		new kernel::discrete::sampled(new kernel::generating::bspline3),
		new kernel::generating::bspline3);
	// no additional discrete processing
	kernel::discrete::base *delta = new kernel::discrete::delta;
	// use mirror extension
	extension::base *ext = new extension::mirror;
	image::rgba<float> rgba0;
	image::load(w,h,image0, &rgba0);
	image::rgba<float> rgba1;
	image::load(w,h,image1, &rgba1);
	image::rgba<float> rgbam0;
	image::load(w, h, mask0, &rgbam0);
	image::rgba<float> rgbam1;
	image::load(w, h, mask1, &rgbam1);

	CPyramidLevel level;
	level.image0 = rgb2gray(w,h,image0);
	level.image1 =  rgb2gray(w,h,image1);
	level.mask0 =  rgb2gray(w,h,mask0);
	level.mask1 = rgb2gray(w,h,mask1);

	
	level.blocks_row = ((w + 4) / 5 + 3) / 4 * 4;
	level.blocks_row += 1 - (level.blocks_row % 2);//make it odd
	level.blocks_col = (h + 4) / 5;
	level.blocks_num = level.blocks_row*level.blocks_col;
	level.w = w;
	level.h = h;
	level.inverse_wh = 1.0f / (level.w*level.h);
	levels.push_back(level);

	int n =  int(log((float)std::min(w,h)) / log(2.0f) - log((float)start_res) / log(2.0f) + 1);
	for (int el = 1; el< n ; el++)
	{
		w = int(floor(w / 2.0f));
		h = int(floor(h / 2.0f));
		unsigned char* temp_image0 = new unsigned char[w*h * 3];
		unsigned char* temp_image1 = new unsigned char[w*h * 3];
		unsigned char* temp_mask0 = new unsigned char[w*h * 3];
		unsigned char* temp_mask1 = new unsigned char[w*h * 3];

		scale(h, w, pre, delta, delta, ext, &rgba0, temp_image0);
		scale(h, w, pre, delta, delta, ext, &rgba1, temp_image1);
		scale(h, w, pre, delta, delta, ext, &rgbam0, temp_mask0);
		scale(h, w, pre, delta, delta, ext, &rgbam1, temp_mask1);

		CPyramidLevel level;
		level.image0 = rgb2gray(w,h,temp_image0);
		level.image1 = rgb2gray(w,h,temp_image1);
		level.mask0 = rgb2gray(w,h,temp_mask0);
		level.mask1 = rgb2gray(w,h,temp_mask1);

		level.blocks_row = ((w + 4) / 5 + 3) / 4 * 4;
		level.blocks_row += 1 - (level.blocks_row % 2);//make it odd
		level.blocks_col = (h + 4) / 5;
		level.blocks_num = level.blocks_row*level.blocks_col;
		level.w = w;
		level.h = h;
		level.inverse_wh = 1.0f / (level.w*level.h);

		levels.push_back(level);

		delete[] temp_image0;
		delete[] temp_image1;
		delete[] temp_mask0;
		delete[] temp_mask1;
	}

	std::cout << "Finished building pramids \n";

	delete pre;
	delete delta;
	delete ext;
}

float* CPyramid::rgb2gray(int w, int h, unsigned char* image)
{
	float* gray = new float[w*h];
	#pragma omp parallel for
	for (int y = 0; y<h; y++)
		for (int x = 0; x < w; x++)
		{
			int index = y*w + x;
			gray[index] = (float)image[index * 3 + 0] * 0.299f + (float)image[index * 3 + 1] * 0.587f + (float)image[index * 3 + 2] * 0.114f;
		
		}

	return gray;
}

