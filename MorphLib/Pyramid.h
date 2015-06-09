#pragma once
#include <vector>

class CPyramidLevel
{
public:
	int w;
	int h;
	float inverse_wh;
	int blocks_row;//how many tiles in a row
	int blocks_col;//how many tiles in a col
	int blocks_num;//total num of tiles
	float* image0;//image1
	float* image1;//image2
	float* mask0;//mask1
	float* mask1;//mask2

	CPyramidLevel()
	{
		image0 = NULL;
		image1 = NULL;
		mask0 = NULL;
		mask1 = NULL;

		w = 0;
		h = 0;
		inverse_wh = 1;
		blocks_row = 0;
		blocks_col = 0;
		blocks_num = 0;


	};

	~CPyramidLevel()
	{
		
	
	};
};

class CPyramid
{
public:
	CPyramid();
	~CPyramid();
	void create_pyramid(int w, int h, unsigned char* image0, unsigned char* image1, unsigned char* mask0, unsigned char* mask1, int start_res);
	float* rgb2gray(int w, int h, unsigned char* image);
	std::vector<CPyramidLevel> levels;	
};
