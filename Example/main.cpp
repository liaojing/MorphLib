//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>  
using namespace cv;

#include <iostream>
#include <fstream>
#include <memory> // unique_ptr<>
using namespace std;

#include "../MorphLib/Header.h"
#include "../MorphLib/Morph.h"

template<class T_in, class T_out>
T_out BilinearGetColor_clamp(cv::Mat& img, float px, float py)//clamp for outside of the boundary
{
	int x[2], y[2];
	T_out value[2][2];
	int w = img.cols;
	int h = img.rows;
	x[0] = int(floor(px));
	y[0] = int(floor(py));
	x[1] = int(ceil(px));
	y[1] = int(ceil(py));

	float u = px - x[0];
	float v = py - y[0];

	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++)
		{
		int temp_x, temp_y;
		temp_x = x[i];
		temp_y = y[j];
		temp_x = std::max(0, temp_x);
		temp_x = std::min(w - 1, temp_x);
		temp_y = std::max(0, temp_y);
		temp_y = std::min(h - 1, temp_y);
		value[i][j] = (img.at<T_in>(temp_y, temp_x));
		}


	return
		value[0][0] * (1 - u)*(1 - v) + value[0][1] * (1 - u)*v + value[1][0] * u*(1 - v) + value[1][1] * u*v;
}


int main(int argc, char *argv[])
{
	//load two images
	Mat image0 = imread("./testcase/image1.png");
	Mat image1 = imread("./testcase/image2.png");
	Mat mask0 = imread("./testcase/mask1.png");
	Mat mask1 = imread("./testcase/mask2.png");

    //compute the halfway parameterization vectors
    Mat vectors(image0.rows, image0.cols, CV_32FC2); {
        // Convert BGRA OpenCV image to RGB array
        auto convert = [](const Mat& mat) -> unique_ptr<unsigned char[]> {
            auto ar = make_unique<unsigned char[]>(mat.rows*mat.cols*3); // RGB
            for (int y = 0; y < mat.rows; y++)
                for (int x = 0; x < mat.cols; x++) {
                    int index = y*mat.cols + x;
                    Vec3b color = mat.at<Vec3b>(y, x); // BGRA
                    ar[index * 3 + 0] = color[2];
                    ar[index * 3 + 1] = color[1];
                    ar[index * 3 + 2] = color[0];
                }
            return ar;
        };
        auto ar_image0 = convert(image0);
        auto ar_image1 = convert(image1);
        auto ar_mask0 = convert(mask0);
        auto ar_mask1 = convert(mask1);

        //set parameters
        MorphLib::CParameters params; {
            params.w = image0.cols;
            params.h = image0.rows;
            params.image0 = ar_image0.get();
            params.image1 = ar_image1.get();
            params.mask0 = ar_mask0.get();
            params.mask1 = ar_mask1.get();
            if (1) { // introduce correspondence constraints
                using CP = MorphLib::ConstraintPoint;
                using MorphLib::Vector2f;
                params.ui_points.push_back(CP(Vector2f(0.273438f, 0.643973f), Vector2f(0.237832f, 0.686947f)));
                params.ui_points.push_back(CP(Vector2f(0.657366f, 0.561384f), Vector2f(0.627212f, 0.448009f)));
                params.ui_points.push_back(CP(Vector2f(0.791295f, 0.641741f), Vector2f(0.755531f, 0.625000f)));
                params.ui_points.push_back(CP(Vector2f(0.768973f, 0.757812f), Vector2f(0.722345f, 0.790929f)));
                params.ui_points.push_back(CP(Vector2f(0.309152f, 0.989955f), Vector2f(0.308628f, 0.990044f)));
                params.ui_points.push_back(CP(Vector2f(0.775442f, 0.994469f), Vector2f(0.775442f, 0.994469f)));
                params.ui_points.push_back(CP(Vector2f(0.373051f, 0.620267f), Vector2f(0.339644f, 0.586860f)));
            }
        }
        //run the matching algorithm
        MorphLib::CMorph morph(params);
        float* vx = new float[params.w*params.h];
        float* vy = new float[params.w*params.h];
        float* ssim_error = new float[params.w*params.h];
        morph.calculate_halfway_parametrization(vx, vy, ssim_error);
        for (int y = 0; y < params.h; y++)
            for (int x = 0; x < params.w; x++) {
                int index = y*params.w + x;
                vectors.at<Vec2f>(y, x) = Vec2f(vx[index], vy[index]);
            }
    }

	//save the halfway parameterization vectors
    if (1) {
        ofstream file("./testcase/vector.txt");
        for (int y = 0; y < vectors.rows; y++)
            for (int x = 0; x < vectors.cols; x++) {
                file << vectors.at<Vec2f>(y, x)[0] <<" "<< vectors.at<Vec2f>(y, x)[1] <<endl;
            }
    }


    //render an intermediate morph image
    if (1) {
        float alpha = 0.5f; // 0.f==image0, 1.f==image1
        cv::Mat image(vectors.rows, vectors.cols, CV_8UC3);
#pragma omp parallel for
        for (int y = 0; y < image.rows; y++)
            for (int x = 0; x < image.cols; x++) {
                Vec2f q {float(x), float(y)};
                Vec2f p = q;
                Vec2f v = vectors.at<Vec2f>(int(p[1]), int(p[0]));
                const float fa = 0.8f; // dampening factor

                for (int i = 0; i < 20; i++) {
                    p = q - (2.f * alpha - 1.f)*v;
                    Vec2f new_v = BilinearGetColor_clamp<Vec2f, Vec2f>(vectors, p[0], p[1]);
                    if (v == new_v)
                        break;
                    v = fa*new_v  + (1.f - fa)*v;
                }

                Vec3f color0 = BilinearGetColor_clamp<Vec3b, Vec3f>(image0, p[0] - v[0], p[1] - v[1]);
                Vec3f color1 = BilinearGetColor_clamp<Vec3b, Vec3f>(image1, p[0] + v[0], p[1] + v[1]);

                image.at<Vec3b>(y, x) = color0*(1.0f-alpha)+color1*alpha;

            }
        imwrite("./testcase/frame050.png", image);
    }
}
