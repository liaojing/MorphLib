#pragma once
#include "Header.h"
#include "Pyramid.h"

typedef struct {
	//ssim
	struct {
		float *cross01, *var0, *var1, *mean0, *mean1, *luma0, *luma1;
		float *value;//old value of ssim
		float* counter;//num of neigbour
	} ssim;

	//tps
	struct {
		float *axy, *bx, *by;

	} tps;

	//ui
	struct
	{
		float *axy, *bx, *by;

	}ui;


	float *mask_ign;

	// vp1
	float *vx;
	float *vy;

	//improving mask
	bool* improving_mask;
	int improving;

}CData;

typedef struct
{
	float x;
	float y;
	int index;
}Pixel;

//3*3 matrix
class NBMatrix
{
public:
	float data[3][3];
	NBMatrix(){ memset(data, 0, 9 * sizeof(float)); }
	NBMatrix(const float in_data)
	{
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				data[y][x] = in_data;
	}

	NBMatrix(const float in_data[3][3])
	{
		memcpy(data, in_data,9 * sizeof(float));
	}

	NBMatrix transpose() {
		NBMatrix A;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				A.data[y][x] = data[4 - y][4 - x];
		return A;
	}


	NBMatrix operator+(const NBMatrix &B) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] + B.data[y][x];
		return C;
	}

	NBMatrix operator+(float v) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] + v;
		return C;
	}

	NBMatrix operator-(const NBMatrix &B) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] - B.data[y][x];
		return C;
	}

	NBMatrix operator-(float v) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] - v;
		return C;
	}

	NBMatrix operator*(const NBMatrix &B) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] * B.data[y][x];
		return C;
	}

	NBMatrix operator*(float v) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] * v;
		return C;
	}

	NBMatrix operator/(const NBMatrix &B) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] / B.data[y][x];
		return C;
	}

	NBMatrix operator/(float v) {
		NBMatrix C;
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				C.data[y][x] = data[y][x] / v;
		return C;
	}

	NBMatrix& operator=(const NBMatrix &B) {
		for (int y = 0; y<3; y++)
			for (int x = 0; x<3; x++)
				data[y][x] = B.data[y][x];
		return *this;
	}

};




class CMorph
{
public:
	CMorph(CParameters &paramsParameters);
	~CMorph();

	const CParameters &params() const;
	CParameters &params();
	bool calculate_halfway_parametrization(float* vx, float* vy, float* ssim_error);
	
private:
	void load_identity(int el);
	template <class FUN>
	void outerloop(int el);
	void initialize_incremental(int el);
	void optimize_highestlevel(int el);
	int optimize_level(int el);
	void upsample_level(int c_el, int n_el);
	template <class CTT, class CT, class CBB, class CB, class FUN>
	void loop_inside_ny(int el, int ny);
	template <class CTT, class CT, class CBB, class CB, class CLL, class CL, class CRR, class CR, class FUN>
	void loop_inside_nx(int el, int ny, int nx);
	template <int ytype, class CNEI, class FUN>
	void loop_inside_by(int el, int nx, int y);
	template<int ytype, int xtype, class CNEI, class FUN>
	void loop_inside_bx(int el, int y, int x);
	template <class CNEI, class COPER>
	void commit_pixel_motion(int el, Pixel& p, CNEI &nei, COPER &op, float dx, float dy);
	template <class CNEI, class COPER>
	float energy_change(int el, Pixel& p, CNEI& nei, COPER& op, float dx, float dy);//dx,dy is the moving vector
	template <class CNEI, class COPER>
	void compute_gradient(int el, Pixel& p, CNEI& nei, COPER& op, float &gx, float &gy);
	template <class CNEI, class COPER>
	float prevent_foldover(int el, Pixel& p, CNEI& nei, COPER& op, float gx, float gy);
	int intersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, float &td, float &d);

	//enengy related
	float ssim(float mean0, float mean1, float var0, float var1, float cross01, float counter);
	template <class CNEI, class COPER>
	void ssim_update(NBMatrix& mean0, NBMatrix& mean1, NBMatrix& var0, NBMatrix& var1, NBMatrix& cross01, NBMatrix& counter, NBMatrix& value, float& luma0, float& luma1, int el, Pixel& p, CNEI& nei, COPER& op, float vx, float vy, float old_mask, float new_mask);


	//tool function
	template <class T>
	inline T BilineaGetColor_clamp(T* img, int w, int h, float px, float py);
	template <class T>
	inline T BilineaGetColor_fill(T* img, int w, int h, float px, float py, T fill);
	inline bool inside(float x, float y, int el);
	inline int mem_index(int x, int y, int el);
	inline float ModMask_ign(float mask1, float mask2);
	



private:
	CParameters& m_params;
	CPyramid *m_pyramid;
	int _total_l, _current_l;
	int _total_iter, _current_iter;
	int _max_iter;
	CData* data;
	bool runflag;

	friend class Opt;
	friend class Init;
};

class Opt
{
public:
	template <class CNEI, class COPER>
	void run(int el, Pixel& p, CNEI& nei, COPER& op, CMorph *match);

	
};

class Init
{
public:
	template <class CNEI, class COPER>
	void run(int el, Pixel& p, CNEI& nei, COPER& op, CMorph *match);
	friend class CMorph;

};

