#include <cstring> 
#include <cctype> 
#include "extension.h"

#include "image.h"
#include "error.h"
static extension::clamp clamp;

namespace image {

	int load(int width, int height, unsigned char *data, image::rgba<float> *rgba) {
		
		if (width != 0 && height != 0)
		{
			rgba->resize(height, width);
			const float tof = (1.f/255.f);
			#pragma omp parallel for
			for (int i = height-1; i >= 0; i--) {
				for (int j = 0; j < width; j++) {
					int p = i*width+j; // flip image so y is up
					rgba->r[p] = color::srgbuncurve(data[p * 3 + 0] * tof);
					rgba->g[p] = color::srgbuncurve(data[p * 3 + 1] * tof);
					rgba->b[p] = color::srgbuncurve(data[p * 3 + 2] * tof);
					rgba->a[p] = 1.0f;
				}
			}
				return 1;
		}
		else
		{
			return 0;
		}       
    }

	int store(unsigned char *data, const image::rgba<float> &rgba) {
        
		int height=rgba.height();
		int width=rgba.width(); 
		
		#pragma omp parallel for
		for (int i = height-1; i >= 0; i--) {
			for (int j = 0; j < width; j++) {
				int p = i*width+j; // flip image so y is up
				float r = color::srgbcurve(clamp(rgba.r[p]));
				float g = color::srgbcurve(clamp(rgba.g[p]));
				float b = color::srgbcurve(clamp(rgba.b[p]));
				
				data[p * 3 + 0] = (unsigned char)(255.f*b);
				data[p * 3 + 1] = (unsigned char)(255.f*g);
				data[p * 3 + 2] = (unsigned char)(255.f*r);
			}
		}
		
        return 1;
    }
}
