//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>  
using namespace cv;

#include <iostream>
#include<fstream>
using namespace std;

#include "..\MorphLib\Header.h"
#include "..\MorphLib\Morph.h"

template<class T_in, class T_out>
T_out BilineaGetColor_clamp(cv::Mat& img, float px, float py)//clamp for outside of the boundary
{
	int x[2], y[2];
	T_out value[2][2];
	int w = img.cols;
	int h = img.rows;
	x[0] = floor(px);
	y[0] = floor(py);
	x[1] = ceil(px);
	y[1] = ceil(py);

	float u = px - x[0];
	float v = py - y[0];

	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++)
		{
		int temp_x, temp_y;
		temp_x = x[i];
		temp_y = y[j];
		temp_x = MAX(0, temp_x);
		temp_x = MIN(w - 1, temp_x);
		temp_y = MAX(0, temp_y);
		temp_y = MIN(h - 1, temp_y);
		value[i][j] = (img.at<T_in>(temp_y, temp_x));
		}


	return
		value[0][0] * (1 - u)*(1 - v) + value[0][1] * (1 - u)*v + value[1][0] * u*(1 - v) + value[1][1] * u*v;
}


int main(int argc, char *argv[])
{

		//load two images
		Mat image0 = imread(".\\testcase\\image1.png");
		Mat image1 = imread(".\\testcase\\image2.png");
		Mat mask0 = imread(".\\testcase\\mask1.png");
		Mat mask1 = imread(".\\testcase\\mask2.png");

		////set parameters
		CParameters params;
		params.w = image0.cols;
		params.h = image1.rows;
		params.image0 = new unsigned char[params.w*params.h * 3];
		params.image1 = new unsigned char[params.w*params.h * 3];
		params.mask0 = new unsigned char[params.w*params.h * 3];
		params.mask1 = new unsigned char[params.w*params.h * 3];
      #pragma omp parallel for
		for (int y = 0; y < params.h; y++)
			for (int x = 0; x < params.w; x++)
			{
			int index = y*params.w + x;
			Vec3b color = image0.at<Vec3b>(y, x);
			params.image0[index * 3 + 0] = color[2];
			params.image0[index * 3 + 1] = color[1];
			params.image0[index * 3 + 2] = color[0];
			color = image1.at<Vec3b>(y, x);
			params.image1[index * 3 + 0] = color[2];
			params.image1[index * 3 + 1] = color[1];
			params.image1[index * 3 + 2] = color[0];
			color = mask0.at<Vec3b>(y, x);
			params.mask0[index * 3 + 0] = color[2];
			params.mask0[index * 3 + 1] = color[1];
			params.mask0[index * 3 + 2] = color[0];
			color = mask1.at<Vec3b>(y, x);
			params.mask1[index * 3 + 0] = color[2];
			params.mask1[index * 3 + 1] = color[1];
			params.mask1[index * 3 + 2] = color[0];

			}
		ConstraintPoint pt;
		pt.lp = Vector2f(0.273438f, 0.643973f);
		pt.rp = Vector2f(0.237832f, 0.686947f);
		params.ui_points.push_back(pt);
		pt.lp = Vector2f(0.657366f, 0.561384f);
		pt.rp = Vector2f(0.627212f, 0.448009f);
		params.ui_points.push_back(pt);
		pt.lp = Vector2f(0.791295f, 0.641741f);
		pt.rp = Vector2f(0.755531f, 0.625000f);
		params.ui_points.push_back(pt);
		pt.lp = Vector2f(0.768973f, 0.757812f);
		pt.rp = Vector2f(0.722345f, 0.790929f);
		params.ui_points.push_back(pt);
		pt.lp = Vector2f(0.309152f, 0.989955f);
		pt.rp = Vector2f(0.308628f, 0.990044f);
		params.ui_points.push_back(pt);
		pt.lp = Vector2f(0.775442f, 0.994469f);
		pt.rp = Vector2f(0.775442f, 0.994469f);
		params.ui_points.push_back(pt);
		pt.lp = Vector2f(0.373051f, 0.620267f);
		pt.rp = Vector2f(0.339644f, 0.586860f);
		params.ui_points.push_back(pt);

		
		//run the matching algorithm
		CMorph morph(params);
		float* vx = new float[params.w*params.h];
		float* vy = new float[params.w*params.h];
		float* ssim_error = new float[params.w*params.h];
		morph.calculate_halfway_parametrization(vx, vy, ssim_error);


		//save the halfway vector
		Mat vectors(params.h, params.w, CV_32FC2);
		ofstream file(".\\testcase\\vector.txt");
		if (file)
		{
			for (int y = 0; y < params.h; y++)
				for (int x = 0; x < params.w; x++)
				{
				int index = y*params.w + x;
				file << vx[index] << " " << vy[index] << endl;
				vectors.at<Vec2f>(y, x) = Vec2f(vx[index], vy[index]);
				}
			file.close();
		}

		delete[] vx;
		delete[] vy;
		delete[] ssim_error;

		
		//rendering
		float alpha = 0.5f;
		cv::Mat image(params.h, params.w, CV_8UC3);
		#pragma omp parallel for
		for (int y = 0; y < params.h; y++)
			for (int x = 0; x < params.w; x++)
			{
			Vec2f q(x, y);
			Vec2f p = q;
			Vec2f v = vectors.at<Vec2f>(p[1], p[0]);
			float fa = 0.8f;

			for (int i = 0; i < 20; i++)
			{
				p = q - (2 * alpha - 1)*v;
				Vec2f new_v = BilineaGetColor_clamp<Vec2f, Vec2f>(vectors, p[0], p[1]);
				if (v == new_v)
					break;
				v = fa*new_v + (1 - fa)*v;
			}


			Vec3f color0;
			Vec3f color1;
			int index = y*params.w + x;

			color0 = BilineaGetColor_clamp<Vec3b, Vec3f>(image0, p[0] - v[0], p[1] - v[1]);
			color1 = BilineaGetColor_clamp<Vec3b, Vec3f>(image1, p[0] + v[0], p[1] + v[1]);


			image.at<Vec3b>(y, x) = color0*(1.0f - alpha) + color1*alpha;

			}
		imwrite(".\\testcase\\frame050.png", image);


}