#include "Morph.h"
#include <iostream>

using namespace Eigen;
using namespace MorphLib;

#define p2(x) (x*x)
#define SIGN(n) (n>0?1:(n<0?-1:0))

class TTW; class TW; class TT; class T;
class BBW; class BW; class BB; class B;
class RRW; class RW; class RR; class R;
class LLW; class LW; class LL; class L;

template <class CTT, class CT, class CBB, class CB,
class CLL, class CL, class CRR, class CR>
class Nei;

template<int ytype, int xtype>
class Oper;


CMorph::CMorph(CParameters &params)
: m_params(params)
{
	if (!m_params.image0 || !m_params.image1)
		throw std::runtime_error("Images are empty");

	if (!m_params.mask0)
	{
		m_params.mask0 = new unsigned char[m_params.w * m_params.h * 3];
		std::memset(m_params.mask0, 0, m_params.w * m_params.h * 3);
	}

	if (!m_params.mask1)
	{
		m_params.mask1 = new unsigned char[m_params.w * m_params.h * 3];
		std::memset(m_params.mask1, 0, m_params.w * m_params.h * 3);
	}


	m_pyramid = new CPyramid();
	try
	{
		m_pyramid->create_pyramid(params.w, params.h, params.image0, params.image1, params.mask0, params.mask1, params.start_res);
	}
	catch (...)
	{
		delete m_pyramid;
		throw;
	}

	runflag = true;
	_total_l = int(m_pyramid->levels.size());
	_current_l = _total_l - 1;

	_total_iter = _current_iter = 0;
	int iter_num = m_params.max_iter;
	for (int el = _total_l - 2; el >= 0; el--)
	{
		iter_num = int(iter_num/m_params.max_iter_drop_factor);
		_total_iter += iter_num*m_pyramid->levels[el].w*m_pyramid->levels[el].h;
	}

	data = (CData*)malloc(_total_l*sizeof(CData));
	for (int el = 0; el<_total_l; el++)
	{
		data[el].ssim.luma0 = NULL;
		data[el].improving_mask = NULL;
		data[el].vx = NULL;
		data[el].vy = NULL;
	}


}


CMorph::~CMorph()
{
	delete m_pyramid;

	for (int el = 0; el<_total_l; el++)
	{
		//all inremental
		if (data[el].ssim.luma0)
			free(data[el].ssim.luma0);

		//improving mask
		if (data[el].improving_mask)
			free(data[el].improving_mask);

		//vector
		if (data[el].vx)
			free(data[el].vx);
		if (data[el].vy)
			free(data[el].vy);
	}
	free(data);
}



bool CMorph::calculate_halfway_parametrization(float* vx, float* vy, float* ssim_error)
{
	clock_t start, finish;

	std::cout << "\n";
	load_identity(_total_l - 1);
	
	if (m_params.ui_points.size() > 0)
	{
		std::cout << "level:" << _total_l - 1 << ", size:" << m_pyramid->levels[_total_l - 1].w << "x" << m_pyramid->levels[_total_l - 1].h << "\n";
		start = clock();
		optimize_highestlevel(_total_l - 1);
		finish = clock();
		float run_time = (float)(finish - start) * 1000 / CLOCKS_PER_SEC;
		std::cout << "time:" << run_time << "ms" << "\n";

	}


	_max_iter = m_params.max_iter;
	for (_current_l = _total_l - 2; _current_l >= 0; _current_l--) {
		_max_iter = int(_max_iter/m_params.max_iter_drop_factor);
		std::cout << "level:" << _current_l << ", size:" << m_pyramid->levels[_current_l].w << "x" << m_pyramid->levels[_current_l].h << "\n";
		start = clock();
		upsample_level(_current_l + 1, _current_l);
		initialize_incremental(_current_l);
		int iter = optimize_level(_current_l);
		finish = clock();
		float run_time = (float)(finish - start) * 1000 / CLOCKS_PER_SEC;
		std::cout << "iter:"<<iter<<", time:" << run_time << "ms" << "\n";

		if (!runflag)
		{
			return false;
		}
	}

	int h = m_pyramid->levels[0].h;
	int w = m_pyramid->levels[0].w;
	
	//update result
	#pragma omp parallel for
	for (int y = 0; y<h; y++)
		for (int x = 0; x<w; x++)
		{
			int index = mem_index(x, y, 0);
			vx[y*w + x] = data[0].vx[index];
			vy[y*w+x] = data[0].vy[index];
			ssim_error[y*w+x] = data[0].ssim.value[index];
		}
	return true;
	
}

void CMorph::upsample_level(int c_el, int n_el)
{
	float ratio_x, ratio_y;
	ratio_x = (float)m_pyramid->levels[n_el].w / (float)m_pyramid->levels[c_el].w;
	ratio_y = (float)m_pyramid->levels[n_el].h / (float)m_pyramid->levels[c_el].h;

	Vector2f* result_c = new Vector2f[m_pyramid->levels[c_el].w*m_pyramid->levels[c_el].h];
	#pragma omp parallel for
	for (int y = 0; y<m_pyramid->levels[c_el].h; y++)
		for (int x = 0; x<m_pyramid->levels[c_el].w; x++)
		{
			int index = mem_index(x, y, c_el);
			float vx = data[c_el].vx[index] * ratio_x;
			float vy = data[c_el].vy[index] * ratio_y;
			result_c[y*m_pyramid->levels[c_el].w+x] = Vector2f(vx, vy);
		}

	
	int size = m_pyramid->levels[n_el].blocks_num * 25 * sizeof(float);
	data[n_el].vx = (float*)malloc(size);
	data[n_el].vy = (float*)malloc(size);
	memset(data[n_el].vx, 0, size);
	memset(data[n_el].vy, 0, size);

	#pragma omp parallel for
	for (int y = 0; y < m_pyramid->levels[n_el].h; y++)
		for (int x = 0; x < m_pyramid->levels[n_el].w; x++)
		{
			float xc = (x + 0.5f) / ratio_x - 0.5f;
			float yc = (y + 0.5f) / ratio_y - 0.5f;

			int index = mem_index(x, y, n_el);
			Vector2f v = BilineaGetColor_clamp<Vector2f>(result_c, m_pyramid->levels[c_el].w, m_pyramid->levels[c_el].h, xc, yc);
			data[n_el].vx[index] = v.x();
			data[n_el].vy[index] = v.y();
		}
	delete[] result_c;
}

void CMorph::load_identity(int el)
{
	int size = m_pyramid->levels[el].blocks_num * 25 * sizeof(float);
	data[el].vx = (float*)malloc(size);
	data[el].vy = (float*)malloc(size);
	memset(data[el].vx, 0, size);
	memset(data[el].vy, 0, size);

}

void CMorph::initialize_incremental(int el)
{

	int size = m_pyramid->levels[el].blocks_num * 25;
	float* incremental_pointer = (float*)malloc(size * 16 * sizeof(float));
	memset(incremental_pointer, 0, size * 16 * sizeof(float));

	//ssim
	data[el].ssim.luma0 = incremental_pointer;
	data[el].ssim.luma1 = incremental_pointer + size * 1;
	data[el].ssim.mean0 = incremental_pointer + size * 2;
	data[el].ssim.mean1 = incremental_pointer + size * 3;
	data[el].ssim.var0 = incremental_pointer + size * 4;
	data[el].ssim.var1 = incremental_pointer + size * 5;
	data[el].ssim.cross01 = incremental_pointer + size * 6;
	data[el].ssim.value = incremental_pointer + size * 7;
	data[el].ssim.counter = incremental_pointer + size * 8;

	//tps
	data[el].tps.axy = incremental_pointer + size * 9;
	data[el].tps.bx = incremental_pointer + size * 10;
	data[el].tps.by = incremental_pointer + size * 11;


	//ui
	data[el].ui.axy = incremental_pointer + size * 12;
	data[el].ui.bx = incremental_pointer + size * 13;
	data[el].ui.by = incremental_pointer + size * 14;

	//mask
	data[el].mask_ign = incremental_pointer + size * 15;

	//improving
	data[el].improving = 1;
	data[el].improving_mask = (bool*)malloc(m_pyramid->levels[el].blocks_num * 25 * sizeof(bool));

	//initialize for ui
	for (size_t i = 0; i<m_params.ui_points.size(); i++)
	{
		float x0 = m_params.ui_points[i].lp.x()*m_pyramid->levels[el].w - 0.5f;
		float y0 = m_params.ui_points[i].lp.y()*m_pyramid->levels[el].h - 0.5f;
		float x1 = m_params.ui_points[i].rp.x()*m_pyramid->levels[el].w - 0.5f;
		float y1 = m_params.ui_points[i].rp.y()*m_pyramid->levels[el].h - 0.5f;
		float con_x = (x0 + x1) / 2.0f;
		float con_y = (y0 + y1) / 2.0f;
		float vx = (x1 - x0) / 2.0f;
		float vy = (y1 - y0) / 2.0f;

		for (int y = int(floor(con_y)); y <= int(ceil(con_y)); y++)
			for (int x = int(floor(con_x)); x <= int(ceil(con_x)); x++)
			{
			if (inside((float)x, (float)y, el))
			{
				int index = mem_index(x, y, el);

				float bilinear_w = (1.0f - fabs(y - con_y))*(1.0f - fabs(x - con_x));

				data[el].ui.axy[index] += bilinear_w;
				data[el].ui.bx[index] += 2.0f*bilinear_w*(data[el].vx[index] - vx);
				data[el].ui.by[index] += 2.0f*bilinear_w*(data[el].vy[index] - vy);
			}
			}
	}

	//initialize each pixel for tps and ssim
	outerloop<Init>(el);
}

int CMorph::optimize_level(int el) {

	
	
	clock_t start, finish;
	start = clock();
	int iter = 0;
	while (data[el].improving>0 && iter<_max_iter) {

		data[el].improving = 0;
		outerloop<Opt>(el);
		iter++;
		_current_iter += m_pyramid->levels[el].w*m_pyramid->levels[el].h;
		if (!runflag)
			break;
	
	}
	finish = clock();
	
	_current_iter += (_max_iter - iter)*m_pyramid->levels[el].w*m_pyramid->levels[el].h;

	return iter;
}

void CMorph::optimize_highestlevel(int el)//linear solver
{
	int w = m_pyramid->levels[el].w;
	int h = m_pyramid->levels[el].h;
	int num = w*h;
	MatrixXf  A(num,num);
	MatrixXf  Bx(num, 1);
	MatrixXf  By(num, 1);
	MatrixXf  X(num, 1);
	MatrixXf  Y(num, 1);
	A.setZero();
	Bx.setZero();
	By.setZero();
	X.setZero();
	Y.setZero();
	
	//set matrixs for tps
#pragma omp parallel for
	for (int y = 0; y<h; y++)
		for (int x = 0; x<w; x++)
		{
		int i = y*w + x;
		//dxx
		if (x>1)
			A(i, i - 2) += 1.0f, A(i, i - 1) += -2.0f, A(i, i) += 1.0f;
		if (x>0 && x<w - 1)
			A(i, i - 1) += -2.0f, A(i, i) += 4.0f, A(i, i + 1) += -2.0f;
		if (x<w - 2)
			A(i,i) += 1.0f, A(i,i+1) += -2.0f, A(i,i+2) += 1.0f;
		//dy
		if (y>1)
			A(i, i - 2 * w) += 1.0f, A(i, i - w) += -2.0f, A(i, i) += 1.0f;
		if (y>0 && y<h - 1)
			A(i, i - w) += -2.0f, A(i, i) += 4.0f, A(i, i + w) += -2.0f;
		if (y<h - 2)
			A(i, i) += 1.0f, A(i, i + w) += -2.0f, A(i, i + 2 * w) += 1.0f;

		//dxy
		if (x>0 && y>0)
			A(i, i - w - 1) += 2.0f, A(i, i - w) += -2.0f, A(i, i - 1) += -2.0f, A(i, i) += 2.0f;
		if (x<w - 1 && y>0)
			A(i, i - w) += -2.0f, A(i, i - w + 1) += 2.0f, A(i, i) += 2.0f, A(i, i + 1) += -2.0f;
		if (x>0 && y<h - 1)
			A(i, i - 1) += -2.0f, A(i, i) += 2.0f, A(i, i + w - 1) += 2.0f, A(i, i + w) += -2.0f;
		if (x<w - 1 && y<h - 1)
			A(i, i) += 2.0f, A(i, i + 1) += -2.0f, A(i, i + w) += -2.0f, A(i, i + w + 1) += 2.0f;

		}

	//set matrix for ui
	for (size_t i = 0; i<m_params.ui_points.size(); i++)
	{
		float x0 = m_params.ui_points[i].lp.x()*m_pyramid->levels[el].w - 0.5f;
		float y0 = m_params.ui_points[i].lp.y()*m_pyramid->levels[el].h - 0.5f;
		float x1 = m_params.ui_points[i].rp.x()*m_pyramid->levels[el].w - 0.5f;
		float y1 = m_params.ui_points[i].rp.y()*m_pyramid->levels[el].h - 0.5f;
		float con_x = (x0 + x1) / 2.0f;
		float con_y = (y0 + y1) / 2.0f;
		float vx = (x1 - x0) / 2.0f;
		float vy = (y1 - y0) / 2.0f;

		for (int y = int(floor(con_y)); y <= int(ceil(con_y)); y++)
			for (int x = int(floor(con_x)); x <= int(ceil(con_x)); x++)
			{
				if (inside(float(x), float(y), el))
				{
					float bilinear_w = (1.0f - fabs(y - con_y))*(1.0f - fabs(x - con_x));
					int i = y*w + x;
					A(i, i) += bilinear_w;
					Bx(i, 0) += bilinear_w*vx;
					By(i, 0) += bilinear_w*vy;				
				}
			}

	}

	//set boundary condistion
	int x, y, i;
	switch (m_params.bcond)
	{
	case BCOND_NONE:
		break;

	case BCOND_CORNER://corner
		x = 0, y = 0;
		i = y*w + x;
		A(i,i) += 10.f;

		x = 0, y = h - 1;
		i = y*w + x;
		A(i, i) += 10.f;

		x = w - 1, y = h - 1;
		i = y*w + x;
		A(i, i) += 10.f;

		x = w - 1, y = 0;
		i = y*w + x;
		A(i, i) += 10.f;
		break;

	case BCOND_BORDER:
		for (x = 0; x<w; x++)
		{
			y = 0;
			i = y*w + x;
			A(i, i) += 10.f;

			y = h - 1;
			i = y*w + x;
			A(i, i) += 10.f;
		}

		for (y = 1; y<h - 1; y++)
		{
			x = 0;
			i = y*w + x;
			A(i, i) += 10.f;

			x = w - 1;
			i = y*w + x;
			A(i, i) += 10.f;

		}

		break;
	}

	FullPivLU<MatrixXf> dec(A);
	X = dec.solve(Bx);
	Y = dec.solve(By);
	
	//load to vx,vy
#pragma omp parallel for
	for (int y = 0; y<h; y++)
	{
		for (int x = 0; x<w; x++)
		{
			int index = mem_index(x, y, el);

			data[el].vx[index] = X(y*w + x, 0);
			data[el].vy[index] = Y(y*w + x, 0);
		}
	}
}



template <class FUN>
void CMorph::outerloop(int el)
{
	loop_inside_ny<TTW, TW, BB, B, FUN>(el, 0);
	loop_inside_ny<TTW, T, BB, B, FUN>(el, 1);
	loop_inside_ny<TT, T, BB, B, FUN>(el, 2);
	loop_inside_ny<TT, T, BBW, B, FUN>(el, 3);
	loop_inside_ny<TT, T, BBW, BW, FUN>(el, 4);
}

template <class CTT, class CT, class CBB, class CB, class FUN>
void CMorph::loop_inside_ny(int el, int ny)
{
	loop_inside_nx<CTT, CT, CBB, CB, LLW, LW, RR, R, FUN>(el, ny, 0);
	loop_inside_nx<CTT, CT, CBB, CB, LLW, L, RR, R, FUN>(el, ny, 1);
	loop_inside_nx<CTT, CT, CBB, CB, LL, L, RR, R, FUN>(el, ny, 2);
	loop_inside_nx<CTT, CT, CBB, CB, LL, L, RRW, R, FUN>(el, ny, 3);
	loop_inside_nx<CTT, CT, CBB, CB, LL, L, RRW, RW, FUN>(el, ny, 4);

}

template <class CTT, class CT, class CBB, class CB, class CLL, class CL, class CRR, class CR, class FUN>
void CMorph::loop_inside_nx(int el, int ny, int nx)
{
	int y = ny;
	for (; y<1; y += 5)
		loop_inside_by<0, Nei<CTT, CT, CBB, CB, CLL, CL, CRR, CR>, FUN>(el, nx, y);

	for (; y<2; y += 5)
		loop_inside_by<1, Nei<CTT, CT, CBB, CB, CLL, CL, CRR, CR>, FUN>(el, nx, y);

	int num = 0;
#pragma omp parallel for
	for (int i = y; i<m_pyramid->levels[el].h - 2; i += 5)
	{
		loop_inside_by<2, Nei<CTT, CT, CBB, CB, CLL, CL, CRR, CR>, FUN>(el, nx, i);
#pragma omp atomic
		num++;
	}
#pragma omp barrier

	for (y += num * 5; y<m_pyramid->levels[el].h - 1; y += 5)
		loop_inside_by<3, Nei<CTT, CT, CBB, CB, CLL, CL, CRR, CR>, FUN>(el, nx, y);

	for (; y<m_pyramid->levels[el].h; y += 5)
		loop_inside_by<4, Nei<CTT, CT, CBB, CB, CLL, CL, CRR, CR>, FUN>(el, nx, y);
}

template <int ytype, class CNEI, class FUN>
void CMorph::loop_inside_by(int el, int nx, int y)
{
	int x = nx;
	for (; x<1; x += 5)
		loop_inside_bx<ytype, 0, CNEI, FUN>(el, y, x);

	for (; x<2; x += 5)
		loop_inside_bx<ytype, 1, CNEI, FUN>(el, y, x);

	for (; x<m_pyramid->levels[el].w - 2; x += 5)
		loop_inside_bx<ytype, 2, CNEI, FUN>(el, y, x);

	for (; x<m_pyramid->levels[el].w - 1; x += 5)
		loop_inside_bx<ytype, 3, CNEI, FUN>(el, y, x);

	for (; x<m_pyramid->levels[el].w; x += 5)
		loop_inside_bx<ytype, 4, CNEI, FUN>(el, y, x);
}

template<int ytype, int xtype, class CNEI, class FUN>
void CMorph::loop_inside_bx(int el, int y, int x)
{
	CNEI nei(m_pyramid->levels[el].blocks_row, m_pyramid->levels[el].blocks_col);
	FUN fun;
	Oper<ytype, xtype> op;

	Pixel p;
	p.x = (float)x; p.y = (float)y;
	p.index = mem_index(x, y, el);

	fun.run(el, p, nei, op, this);

};

//Gradient descent
template <class CNEI, class COPER>
void Opt::run(int el, Pixel& p, CNEI& nei, COPER& op, CMorph* match)
{
	int w = match->m_pyramid->levels[el].w;
	int h = match->m_pyramid->levels[el].h;
	if (!op.io.flag(p.index, nei, match->data[el].improving_mask))
		return;
	match->data[el].improving_mask[p.index] = false;

	float gx, gy;
	match->compute_gradient(el, p, nei, op, gx, gy);
	float ng = sqrt(p2(gx) + p2(gy));
	if (ng == 0.0f)
		return;
	gx /= ng;
	gy /= ng;

	//different boundary condition
	switch (match->m_params.bcond)
	{
	case BCOND_NONE:
		break;
	case BCOND_CORNER:
		if ((p.x == 0 && p.y == 0) || (p.x == 0 && p.y == h - 1) || (p.x == w - 1 && p.y == 0) || (p.x == w - 1 && p.y == h - 1))
			return;
		break;
	case BCOND_BORDER:
		if (p.x == 0 || p.y == 0 || p.y == h - 1 || p.x == w - 1)
			return;
		break;
		/*
		case BCOND_RECT:
		if(p.x==0||p.x==w-1)
		gx=0;
		if(p.y==0||p.y==h-1)
		gy=0;
		break;
		*/
	}


	//golden section
	float t0 = 0.0f;
	float t3 = match->prevent_foldover(el, p, nei, op, gx, gy);
	const float R = 0.618033989f;
	const float C = 1.0f - R;
	float t1 = R*t0 + C*t3;
	float t2 = R*t1 + C*t3;
	float f1 = match->energy_change(el, p, nei, op, gx*t1, gy*t1);
	float f2 = match->energy_change(el, p, nei, op, gx*t2, gy*t2);

	while (t3 - t0 > match->m_params.eps) {
		if (f2<f1)
		{
			t0 = t1;
			t1 = t2;
			t2 = R*t2 + C*t3;
			f1 = f2;
			f2 = match->energy_change(el, p, nei, op, gx*t2, gy*t2);
		}
		else
		{
			t3 = t2;
			t2 = t1;
			t1 = R*t1 + C*t0;
			f2 = f1;
			f1 = match->energy_change(el, p, nei, op, gx*t1, gy*t1);
		}

	}

	float tmin, fmin;
	if (f1<f2)
		tmin = t1, fmin = f1;
	else
		tmin = t2, fmin = f2;


	// commit changes?
	float dx = gx*tmin, dy = gy*tmin;
	if (fmin < 0.f) 	{
		match->commit_pixel_motion(el, p, nei, op, dx, dy);
		match->data[el].improving++;
	}
}



template <class CNEI, class COPER>
void Init::run(int el, Pixel& p, CNEI& nei, COPER& op, CMorph* match)
{
	NBMatrix vx;//Motion vector
	NBMatrix vy;//Motion vector

	op.io.readNB(p.index, nei, vx, match->data[el].vx);
	op.io.readNB(p.index, nei, vy, match->data[el].vy);

	//ssim
	NBMatrix luma0, luma1;
	NBMatrix mask;
	for (int y = 0; y<3; y++)
		for (int x = 0; x<3; x++)
		{
		luma0.data[y][x] = match->BilineaGetColor_clamp(match->m_pyramid->levels[el].image0, match->m_pyramid->levels[el].w, match->m_pyramid->levels[el].h,p.x + x - 1 - vx.data[y][x], p.y + y - 1 - vy.data[y][x]);
		luma1.data[y][x] = match->BilineaGetColor_clamp(match->m_pyramid->levels[el].image1, match->m_pyramid->levels[el].w, match->m_pyramid->levels[el].h, p.x + x - 1 + vx.data[y][x], p.y + y - 1 + vy.data[y][x]);
		float mask0 = match->BilineaGetColor_fill(match->m_pyramid->levels[el].mask0, match->m_pyramid->levels[el].w, match->m_pyramid->levels[el].h, p.x + x - 1 - vx.data[y][x], p.y + y - 1 - vy.data[y][x], 255.0f);
		float mask1 = match->BilineaGetColor_fill(match->m_pyramid->levels[el].mask1, match->m_pyramid->levels[el].w, match->m_pyramid->levels[el].h, p.x + x - 1 + vx.data[y][x], p.y + y - 1 + vy.data[y][x], 255.0f);
		mask.data[y][x] = match->ModMask_ign(mask0, mask1);
		}

	match->data[el].ssim.luma0[p.index] = luma0.data[1][1];
	match->data[el].ssim.luma1[p.index] = luma1.data[1][1];
	match->data[el].mask_ign[p.index] = mask.data[1][1];

	float counter = match->data[el].ssim.counter[p.index] = op.io.sumNB(mask);
	float m0 = match->data[el].ssim.mean0[p.index] = op.io.sumNB(luma0*mask);
	float m1 = match->data[el].ssim.mean1[p.index] = op.io.sumNB(luma1*mask);

	NBMatrix var0, var1;
	var0 = luma0*luma0;
	var1 = luma1*luma1;
	float v0 = match->data[el].ssim.var0[p.index] = op.io.sumNB(var0*mask);
	float v1 = match->data[el].ssim.var1[p.index] = op.io.sumNB(var1*mask);

	NBMatrix cross01;
	cross01 = luma0*luma1;
	float cr01 = match->data[el].ssim.cross01[p.index] = op.io.sumNB(cross01*mask);

	match->data[el].ssim.value[p.index] = match->ssim(m0, m1, v0, v1, cr01, counter);

	//TPS
	match->data[el].tps.axy[p.index] = op.tps.stencil[2][2] / 2;
	match->data[el].tps.bx[p.index] = op.tps.get(p.index, nei, match->data[el].vx);
	match->data[el].tps.by[p.index] = op.tps.get(p.index, nei, match->data[el].vy);

	//improving map
	match->data[el].improving_mask[p.index] = true;//allow moving
}

template <class CNEI, class COPER>
void CMorph::compute_gradient(int el, Pixel& p, CNEI& nei, COPER& op, float &gx, float &gy) {

	gx = -(energy_change(el, p, nei, op, m_params.eps, 0.0f) - energy_change(el, p, nei, op, -m_params.eps, 0.0f));
	gy = -(energy_change(el, p, nei, op, 0.0f, m_params.eps) - energy_change(el, p, nei, op, 0.0f, -m_params.eps));

}

template <class CNEI, class COPER>
float CMorph::prevent_foldover(int el, Pixel& p, CNEI& nei, COPER& op, float gx, float gy)
{
	//image
	float cx0, cy0, cx1, cy1;
	cx0 = p.x - data[el].vx[p.index];
	cy0 = p.y - data[el].vy[p.index];
	cx1 = p.x + data[el].vx[p.index];
	cy1 = p.y + data[el].vy[p.index];

	float nx0[8], ny0[8], nx1[8], ny1[8];
	op.io.oneringNB(p, nei, nx0, ny0, nx1, ny1, data[el].vx, data[el].vy);

	float td[16], d[16];
	int inter_num = 0;
	for (int i = 0; i<8; i++)
	{
		if (intersect(cx0, cy0, cx0 - gx, cy0 - gy, nx0[i], ny0[i], nx0[(i + 1) % 8], ny0[(i + 1) % 8], td[inter_num], d[inter_num]))
			inter_num++;
		if (intersect(cx1, cy1, cx1 + gx, cy1 + gy, nx1[i], ny1[i], nx1[(i + 1) % 8], ny1[(i + 1) % 8], td[inter_num], d[inter_num]))
			inter_num++;
	}

	//find the smallest non-negative t
	if (inter_num == 0)
		return 1.0f;

	float td_min = 1.0f, d_min = 0.0f;
	for (int i = 0; i<inter_num; i++)
	{
		if (td[i] >= 0 && td[i] * d_min<d[i] * td_min)
			td_min = td[i], d_min = d[i];
	}

	if (fabs(d_min)>0.00001f)
		return std::max(td_min / d_min - m_params.eps, 0.0f);
	else
		return 1.0f;
}



int CMorph::intersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, float &td, float &d)
{
	d = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1);
	td = (x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3);
	float ud = (x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3);

	if (d > 0) {
		if (ud >= 0 && ud <= d)
			return 1;
		else
			return 0;
	}
	else {
		if (ud <= 0 && ud >= d) {
			td = -td;
			d = -d;
			return 1;
		}
		else
			return 0;
	}
}


template <class CNEI, class COPER>
float CMorph::energy_change(int el, Pixel& p, CNEI& nei, COPER& op, float dx, float dy)//dx,dy is the moving vector
{
	float vx = data[el].vx[p.index];
	float vy = data[el].vy[p.index];

	//ign_mask
	float old_mask, new_mask;
	old_mask = data[el].mask_ign[p.index];
	float mask0 = BilineaGetColor_fill(m_pyramid->levels[el].mask0, m_pyramid->levels[el].w, m_pyramid->levels[el].h, p.x - vx - dx, p.y - vy - dy, 255.0f);
	float mask1 = BilineaGetColor_fill(m_pyramid->levels[el].mask1, m_pyramid->levels[el].w, m_pyramid->levels[el].h, p.x + vx + dx, p.y + vy + dy, 255.0f);
	new_mask = ModMask_ign(mask0, mask1);

	//ssim
	NBMatrix mean0, mean1, var0, var1, cross01, counter, new_ssim, old_ssim;
	op.io.readNB(p.index, nei, old_ssim, data[el].ssim.value);
	float luma0, luma1;
	ssim_update(mean0, mean1, var0, var1, cross01, counter, new_ssim, luma0, luma1, el, p, nei, op, vx + dx, vy + dy, old_mask, new_mask);
	float v_ssim = op.io.sumNB(new_ssim - old_ssim);

	//tps
	float v_tps = 0.0f;
	v_tps += data[el].tps.axy[p.index] * p2(dx);
	v_tps += data[el].tps.axy[p.index] * p2(dy);
	v_tps += data[el].tps.bx[p.index] * dx;
	v_tps += data[el].tps.by[p.index] * dy;


	//ui
	float v_ui = 0.0f;
	v_ui += data[el].ui.axy[p.index] * p2(dx);
	v_ui += data[el].ui.axy[p.index] * p2(dy);
	v_ui += data[el].ui.bx[p.index] * dx;
	v_ui += data[el].ui.by[p.index] * dy;

	return (m_params.w_ui*v_ui + m_params.w_ssim*v_ssim)*m_pyramid->levels[el].inverse_wh + m_params.w_tps*v_tps;

}

template <class CNEI, class COPER>
void CMorph::commit_pixel_motion(int el, Pixel& p, CNEI &nei, COPER &op, float dx, float dy)
{
	float vx = data[el].vx[p.index];
	float vy = data[el].vy[p.index];

	//ign_mask
	float old_mask, new_mask;
	old_mask = data[el].mask_ign[p.index];
	float mask0 = BilineaGetColor_fill(m_pyramid->levels[el].mask0, m_pyramid->levels[el].w, m_pyramid->levels[el].h, p.x - vx - dx, p.y - vy - dy, 255.0f);
	float mask1 = BilineaGetColor_fill(m_pyramid->levels[el].mask1, m_pyramid->levels[el].w, m_pyramid->levels[el].h, p.x + vx + dx, p.y + vy + dy, 255.0f);
	new_mask = data[el].mask_ign[p.index] = ModMask_ign(mask0, mask1);

	//ssim
	NBMatrix mean0, mean1, var0, var1, cross01, counter, value;
	float luma0, luma1;
	ssim_update(mean0, mean1, var0, var1, cross01, counter, value, luma0, luma1, el, p, nei, op, vx + dx, vy + dy, old_mask, new_mask);
	op.io.writeNB(p.index, nei, mean0, data[el].ssim.mean0);
	op.io.writeNB(p.index, nei, mean1, data[el].ssim.mean1);
	op.io.writeNB(p.index, nei, var0, data[el].ssim.var0);
	op.io.writeNB(p.index, nei, var1, data[el].ssim.var1);
	op.io.writeNB(p.index, nei, cross01, data[el].ssim.cross01);
	op.io.writeNB(p.index, nei, counter, data[el].ssim.counter);
	op.io.writeNB(p.index, nei, value, data[el].ssim.value);
	data[el].ssim.luma0[p.index] = luma0;
	data[el].ssim.luma1[p.index] = luma1;

	//tps
	op.tps.update(p.index, nei, dx, data[el].tps.bx);
	op.tps.update(p.index, nei, dy, data[el].tps.by);

	//ui
	data[el].ui.bx[p.index] += 2.0f*dx*data[el].ui.axy[p.index];
	data[el].ui.by[p.index] += 2.0f*dy*data[el].ui.axy[p.index];

	//vector
	data[el].vx[p.index] += dx;
	data[el].vy[p.index] += dy;

	//imprving mask
	data[el].improving_mask[p.index] = true;

}


float CMorph::ssim(float mean0, float mean1, float var0, float var1, float cross01, float counter)
{
	if (counter <= 1.0f) return 0.0f;
	//m_params
	const float c2 = 58.5225f;
	const float c3 = 29.26125f;

	mean0 /= counter;
	mean1 /= counter;

	var0 = (var0 - counter*mean0*mean0) / counter;
	var0 = fmax(0.0f, var0);

	var1 = (var1 - counter*mean1*mean1) / counter;
	var1 = fmax(0.0f, var1);

	cross01 = (cross01 - counter*mean0*mean1) / counter;

	float var0_root = sqrt(var0);
	float var1_root = sqrt(var1);
	float c = (2 * var0_root*var1_root + c2) / (var0 + var1 + c2);
	float s = (fabs(cross01) + c3) / (var0_root*var1_root + c3);

	return std::min(std::max(0.0f, 1.0f - c*s), 1.0f - m_params.ssim_clamp);
}

template <class CNEI, class COPER>
void CMorph::ssim_update(NBMatrix& mean0, NBMatrix& mean1, NBMatrix& var0, NBMatrix& var1, NBMatrix& cross01, NBMatrix& counter, NBMatrix& value, float& luma0, float& luma1, int el, Pixel& p, CNEI& nei, COPER& op, float vx, float vy, float old_mask, float new_mask)
{
	//ssim
	float old_luma0 = data[el].ssim.luma0[p.index];
	float old_luma1 = data[el].ssim.luma1[p.index];
	luma0 = BilineaGetColor_clamp(m_pyramid->levels[el].image0, m_pyramid->levels[el].w, m_pyramid->levels[el].h, p.x - vx, p.y - vy);
	luma1 = BilineaGetColor_clamp(m_pyramid->levels[el].image1, m_pyramid->levels[el].w, m_pyramid->levels[el].h, p.x + vx, p.y + vy);


	op.io.readNB(p.index, nei, mean0, data[el].ssim.mean0);
	op.io.readNB(p.index, nei, mean1, data[el].ssim.mean1);
	op.io.readNB(p.index, nei, var0, data[el].ssim.var0);
	op.io.readNB(p.index, nei, var1, data[el].ssim.var1);
	op.io.readNB(p.index, nei, cross01, data[el].ssim.cross01);
	op.io.readNB(p.index, nei, counter, data[el].ssim.counter);

	mean0 = mean0 + (luma0*new_mask - old_luma0*old_mask);
	mean1 = mean1 + (luma1*new_mask - old_luma1*old_mask);
	var0 = var0 + (p2(luma0)*new_mask - p2(old_luma0)*old_mask);
	var1 = var1 + (p2(luma1)*new_mask - p2(old_luma1)*old_mask);
	cross01 = cross01 + (luma0*luma1*new_mask - old_luma0*old_luma1*old_mask);
	counter = counter + (new_mask - old_mask);

	for (int y = 0; y<3; y++)
		for (int x = 0; x<3; x++)
		{
		value.data[y][x] = ssim(mean0.data[y][x], mean1.data[y][x], var0.data[y][x], var1.data[y][x], cross01.data[y][x], counter.data[y][x]);
		}

}

////inline functions////
template <class T>
inline T CMorph::BilineaGetColor_clamp(T* img, int w, int h, float px, float py)
{
	int x[2], y[2];
	T value[2][2];
	
	x[0] = (int)floor(px);
	y[0] = (int)floor(py);
	x[1] = (int)ceil(px);
	y[1] = (int)ceil(py);

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
		value[i][j] = img[temp_y*w+temp_x];
		}


	return
		value[0][0] * (1 - u)*(1 - v) + value[0][1] * (1 - u)*v + value[1][0] * u*(1 - v) + value[1][1] * u*v;
}

template <class T>
T CMorph::BilineaGetColor_fill(T* img, int w, int h, float px, float py, T fill)//clamp for outside of the boundary
{
	int x[2], y[2];
	T value[2][2];
	
	x[0] = (int)floor(px);
	y[0] = (int)floor(py);
	x[1] = (int)ceil(px);
	y[1] = (int)ceil(py);

	float u = px - x[0];
	float v = py - y[0];

	for (int i = 0; i<2; i++)
		for (int j = 0; j<2; j++)
		{
			int temp_x, temp_y;
			temp_x = x[i];
			temp_y = y[j];
			if (temp_x<0 || temp_x>w - 1 || temp_y<0 || temp_y>h - 1)
				value[i][j] = fill;
			else
				value[i][j] = img[temp_y*w + temp_x];
		}


	return
		value[0][0] * (1 - u)*(1 - v) + value[0][1] * (1 - u)*v + value[1][0] * u*(1 - v) + value[1][1] * u*v;
}
inline float CMorph::ModMask_ign(float mask0, float mask1)
{

	return (1.0f - mask0 / 255.0f)*(1.0f - mask1 / 255.0f);

}


inline bool CMorph::inside(float x, float y, int el)
{
	if (y >= 0.0f&&y<m_pyramid->levels[el].h&&x >= 0.0f&&x<m_pyramid->levels[el].w)
		return true;
	else
		return false;
}

inline int CMorph::mem_index(int x, int y, int el) {
	int color_index = y % 5 * 5 + x % 5;
	int block_index = y / 5 * m_pyramid->levels[el].blocks_row + x / 5;
	return color_index*m_pyramid->levels[el].blocks_num + block_index;
}

//class as function

//get neighbor
class TT
{
public:
	TT(int row, int col)
	{
		_offset = -row*col * 10;
	}

	int operator()(int index) const {
		return index + _offset;
	}


private:
	int _offset;
}
;

class TTW
{
public:
	TTW(int row, int col)
	{
		_offset = (row*col) * 15 - row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};

class T
{
public:
	T(int row, int col)
	{
		_offset = -row*col * 5;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class TW
{
public:
	TW(int row, int col)
	{
		_offset = row*col * 20 - row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class BB
{
public:
	BB(int row, int col)
	{
		_offset = row*col * 10;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class BBW
{
public:
	BBW(int row, int col)
	{
		_offset = -row*col * 15 + row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class B
{
public:
	B(int row, int col)
	{
		_offset = row*col * 5;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class BW
{
public:
	BW(int row, int col)
	{
		_offset = -row*col * 20 + row;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class LL
{
public:
	LL(int row, int col)
	{
		_offset = -2 * row*col;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class LLW
{
public:
	LLW(int row, int col)
	{
		_offset = row*col * 3 - 1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class L
{
public:
	L(int row, int col)
	{
		_offset = -row*col;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class LW
{
public:
	LW(int row, int col)
	{
		_offset = row*col * 4 - 1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class RR
{
public:
	RR(int row, int col)
	{
		_offset = row*col * 2;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class RRW
{
public:
	RRW(int row, int col)
	{
		_offset = -row*col * 3 + 1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};



class R
{
public:
	R(int row, int col)
	{
		_offset = row*col;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


class RW
{
public:
	RW(int row, int col)
	{
		_offset = -row*col * 4 + 1;
	}

	int operator()(int index) const {
		return index + _offset;
	}

private:
	int _offset;
};


template <class CTT, class CT, class CBB, class CB, class CLL, class CL, class CRR, class CR>
class Nei
{
public:
	CTT tt;
	CT t;
	CBB bb;
	CB b;
	CLL ll;
	CL l;
	CRR rr;
	CR r;

	Nei(int row, int col) :tt(row, col), t(row, col), bb(row, col), b(row, col), ll(row, col), l(row, col), rr(row, col), r(row, col)
	{
	}
};



template<int ytype, int xtype>
class IO
{
public:
	static const int stencil[5][5];
	static const int counter;

	template <class CNEI>
	void readNB(int index, CNEI &nei, NBMatrix &matrix, float* src_data){
		if (stencil[1][1] != 0)	matrix.data[0][0] = src_data[nei.l(nei.t(index))];
		if (stencil[1][2] != 0)	matrix.data[0][1] = src_data[nei.t(index)];
		if (stencil[1][3] != 0)	matrix.data[0][2] = src_data[nei.r(nei.t(index))];
		if (stencil[2][1] != 0)	matrix.data[1][0] = src_data[nei.l(index)];
		if (stencil[2][2] != 0)	matrix.data[1][1] = src_data[index];
		if (stencil[2][3] != 0)	matrix.data[1][2] = src_data[nei.r(index)];
		if (stencil[3][1] != 0)	matrix.data[2][0] = src_data[nei.l(nei.b(index))];
		if (stencil[3][2] != 0)	matrix.data[2][1] = src_data[nei.b(index)];
		if (stencil[3][3] != 0)	matrix.data[2][2] = src_data[nei.r(nei.b(index))];
		
	}

	template <class CNEI>
	void writeNB(int index, CNEI &nei, NBMatrix &matrix, float* dst_data)
	{
		
		if (stencil[1][1] != 0)	dst_data[nei.l(nei.t(index))] = matrix.data[0][0];
		if (stencil[1][2] != 0)	dst_data[nei.t(index)] = matrix.data[0][1];
		if (stencil[1][3] != 0)	dst_data[nei.r(nei.t(index))] = matrix.data[0][2];
		if (stencil[2][1] != 0)	dst_data[nei.l(index)] = matrix.data[1][0];
		if (stencil[2][2] != 0)	dst_data[index] = matrix.data[1][1];
		if (stencil[2][3] != 0)	dst_data[nei.r(index)] = matrix.data[1][2];
		if (stencil[3][1] != 0)	dst_data[nei.l(nei.b(index))] = matrix.data[2][0];
		if (stencil[3][2] != 0)	dst_data[nei.b(index)] = matrix.data[2][1];
		if (stencil[3][3] != 0)	dst_data[nei.r(nei.b(index))] = matrix.data[2][2];
		
	}



	float sumNB(const NBMatrix& matrix)
	{
		float sum = 0.0f;
		
		if (stencil[1][1] != 0)	sum += matrix.data[0][0];
		if (stencil[1][2] != 0)	sum += matrix.data[0][1];
		if (stencil[1][3] != 0)	sum += matrix.data[0][2];
		if (stencil[2][1] != 0)	sum += matrix.data[1][0];
		if (stencil[2][2] != 0)	sum += matrix.data[1][1];
		if (stencil[2][3] != 0)	sum += matrix.data[1][2];
		if (stencil[3][1] != 0)	sum += matrix.data[2][0];
		if (stencil[3][2] != 0)	sum += matrix.data[2][1];
		if (stencil[3][3] != 0)	sum += matrix.data[2][2];
		
		return sum;
	}

	template <class CNEI>
	bool flag(int index, CNEI nei, bool *src_data)
	{
		if (stencil[0][0] != 0)	{ if (src_data[nei.ll(nei.tt(index))]) return true; }
		if (stencil[0][1] != 0)	{ if (src_data[nei.l(nei.tt(index))]) return true; }
		if (stencil[0][2] != 0)	{ if (src_data[nei.tt(index)]) return true; }
		if (stencil[0][3] != 0)	{ if (src_data[nei.r(nei.tt(index))]) return true; }
		if (stencil[0][4] != 0)	{ if (src_data[nei.rr(nei.tt(index))]) return true; }
		if (stencil[1][0] != 0)	{ if (src_data[nei.ll(nei.t(index))]) return true; }
		if (stencil[1][1] != 0)	{ if (src_data[nei.l(nei.t(index))]) return true; }
		if (stencil[1][2] != 0)	{ if (src_data[nei.t(index)]) return true; }
		if (stencil[1][3] != 0)	{ if (src_data[nei.r(nei.t(index))]) return true; }
		if (stencil[1][4] != 0)	{ if (src_data[nei.rr(nei.t(index))]) return true; }
		if (stencil[2][0] != 0)	{ if (src_data[nei.ll(index)]) return true; }
		if (stencil[2][1] != 0)	{ if (src_data[nei.l(index)]) return true; }
		if (stencil[2][2] != 0)	{ if (src_data[index]) return true; }
		if (stencil[2][3] != 0)	{ if (src_data[nei.r(index)]) return true; }
		if (stencil[2][4] != 0)	{ if (src_data[nei.rr(index)]) return true; }
		if (stencil[3][0] != 0)	{ if (src_data[nei.ll(nei.b(index))]) return true; }
		if (stencil[3][1] != 0)	{ if (src_data[nei.l(nei.b(index))]) return true; }
		if (stencil[3][2] != 0)	{ if (src_data[nei.b(index)]) return true; }
		if (stencil[3][3] != 0)	{ if (src_data[nei.r(nei.b(index))]) return true; }
		if (stencil[3][4] != 0)	{ if (src_data[nei.rr(nei.b(index))]) return true; }
		if (stencil[4][0] != 0)	{ if (src_data[nei.ll(nei.bb(index))]) return true; }
		if (stencil[4][1] != 0)	{ if (src_data[nei.l(nei.bb(index))]) return true; }
		if (stencil[4][2] != 0)	{ if (src_data[nei.bb(index)]) return true; }
		if (stencil[4][3] != 0)	{ if (src_data[nei.r(nei.bb(index))]) return true; }
		if (stencil[4][4] != 0)	{ if (src_data[nei.rr(nei.bb(index))]) return true; }

		return false;
	}
	template <class CNEI>
	void oneringNB(Pixel &p, CNEI &nei, float nx0[8], float ny0[8], float nx1[8], float ny1[8], float *vx, float* vy)//in clockwise order
	{
		int index = p.index;
		float cx0, cy0, cx1, cy1;
		cx0 = p.x - vx[index];
		cy0 = p.y - vy[index];
		cx1 = p.x + vx[index];
		cy1 = p.y + vy[index];

		if (stencil[1][1] != 0)
		{
			nx0[0] = p.x - 1.0f - vx[nei.l(nei.t(index))];
			ny0[0] = p.y - 1.0f - vy[nei.l(nei.t(index))];
			nx1[0] = p.x - 1.0f + vx[nei.l(nei.t(index))];
			ny1[0] = p.y - 1.0f + vy[nei.l(nei.t(index))];
		}
		else
		{
			nx0[0] = cx0 - 1.0f;
			ny0[0] = cy0 - 1.0f;
			nx1[0] = cx1 - 1.0f;
			ny1[0] = cy1 - 1.0f;
		}

		if (stencil[1][2] != 0)
		{
			nx0[1] = p.x - vx[nei.t(index)];
			ny0[1] = p.y - 1.0f - vy[nei.t(index)];
			nx1[1] = p.x + vx[nei.t(index)];
			ny1[1] = p.y - 1.0f + vy[nei.t(index)];
		}
		else
		{
			nx0[1] = cx0;
			ny0[1] = cy0 - 1.0f;
			nx1[1] = cx1;
			ny1[1] = cy1 - 1.0f;
		}

		if (stencil[1][3] != 0)
		{
			nx0[2] = p.x + 1.0f - vx[nei.r(nei.t(index))];
			ny0[2] = p.y - 1.0f - vy[nei.r(nei.t(index))];
			nx1[2] = p.x + 1.0f + vx[nei.r(nei.t(index))];
			ny1[2] = p.y - 1.0f + vy[nei.r(nei.t(index))];
		}
		else
		{
			nx0[2] = cx0 + 1.0f;
			ny0[2] = cy0 - 1.0f;
			nx1[2] = cx1 + 1.0f;
			ny1[2] = cy1 - 1.0f;
		}


		if (stencil[2][3] != 0)
		{
			nx0[3] = p.x + 1.0f - vx[nei.r(index)];
			ny0[3] = p.y - vy[nei.r(index)];
			nx1[3] = p.x + 1.0f + vx[nei.r(index)];
			ny1[3] = p.y + vy[nei.r(index)];
		}
		else
		{
			nx0[3] = cx0 + 1.0f;
			ny0[3] = cy0;
			nx1[3] = cx1 + 1.0f;
			ny1[3] = cy1;
		}


		if (stencil[3][3] != 0)
		{
			nx0[4] = p.x + 1.0f - vx[nei.r(nei.b(index))];
			ny0[4] = p.y + 1.0f - vy[nei.r(nei.b(index))];
			nx1[4] = p.x + 1.0f + vx[nei.r(nei.b(index))];
			ny1[4] = p.y + 1.0f + vy[nei.r(nei.b(index))];
		}
		else
		{
			nx0[4] = cx0 + 1.0f;
			ny0[4] = cy0 + 1.0f;
			nx1[4] = cx1 + 1.0f;
			ny1[4] = cy1 + 1.0f;
		}

		if (stencil[3][2] != 0)
		{
			nx0[5] = p.x - vx[nei.b(index)];
			ny0[5] = p.y + 1.0f - vy[nei.b(index)];
			nx1[5] = p.x + vx[nei.b(index)];
			ny1[5] = p.y + 1.0f + vy[nei.b(index)];
		}
		else
		{
			nx0[5] = cx0;
			ny0[5] = cy0 + 1.0f;
			nx1[5] = cx1;
			ny1[5] = cy1 + 1.0f;
		}

		if (stencil[3][1] != 0)
		{
			nx0[6] = p.x - 1.0f - vx[nei.l(nei.b(index))];
			ny0[6] = p.y + 1.0f - vy[nei.l(nei.b(index))];
			nx1[6] = p.x - 1.0f + vx[nei.l(nei.b(index))];
			ny1[6] = p.y + 1.0f + vy[nei.l(nei.b(index))];
		}
		else
		{
			nx0[6] = cx0 - 1.0f;
			ny0[6] = cy0 + 1.0f;
			nx1[6] = cx1 - 1.0f;
			ny1[6] = cy1 + 1.0f;
		}

		if (stencil[2][1] != 0)
		{
			nx0[7] = p.x - 1.0f - vx[nei.l(index)];
			ny0[7] = p.y - vy[nei.l(index)];
			nx1[7] = p.x - 1.0f + vx[nei.l(index)];
			ny1[7] = p.y + vy[nei.l(index)];
		}
		else
		{
			nx0[7] = cx0 - 1.0f;
			ny0[7] = cy0;
			nx1[7] = cx1 - 1.0f;
			ny1[7] = cy1;
		}
	}
};

template<> const int IO<0, 0>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 } };
template<> const int IO<0, 1>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 } };
template<> const int IO<0, 2>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 } };
template<> const int IO<0, 3>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 } };
template<> const int IO<0, 4>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 } };
template<> const int IO<1, 0>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 } };
template<> const int IO<1, 1>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 } };
template<> const int IO<1, 2>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 } };
template<> const int IO<1, 3>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 } };
template<> const int IO<1, 4>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 } };
template<> const int IO<2, 0>::stencil[5][5] = { { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 } };
template<> const int IO<2, 1>::stencil[5][5] = { { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 } };
template<> const int IO<2, 2>::stencil[5][5] = { { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 } };
template<> const int IO<2, 3>::stencil[5][5] = { { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 } };
template<> const int IO<2, 4>::stencil[5][5] = { { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 } };
template<> const int IO<3, 0>::stencil[5][5] = { { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<3, 1>::stencil[5][5] = { { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<3, 2>::stencil[5][5] = { { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<3, 3>::stencil[5][5] = { { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<3, 4>::stencil[5][5] = { { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<4, 0>::stencil[5][5] = { { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<4, 1>::stencil[5][5] = { { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<4, 2>::stencil[5][5] = { { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<4, 3>::stencil[5][5] = { { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const int IO<4, 4>::stencil[5][5] = { { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 1, 1, 1, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };


template<> const int IO<0, 0>::counter = 9;
template<> const int IO<0, 1>::counter = 12;
template<> const int IO<0, 2>::counter = 15;
template<> const int IO<0, 3>::counter = 12;
template<> const int IO<0, 4>::counter = 9;
template<> const int IO<1, 0>::counter = 12;
template<> const int IO<1, 1>::counter = 16;
template<> const int IO<1, 2>::counter = 20;
template<> const int IO<1, 3>::counter = 16;
template<> const int IO<1, 4>::counter = 12;
template<> const int IO<2, 0>::counter = 15;
template<> const int IO<2, 1>::counter = 20;
template<> const int IO<2, 2>::counter = 25;
template<> const int IO<2, 3>::counter = 20;
template<> const int IO<2, 4>::counter = 15;
template<> const int IO<3, 0>::counter = 12;
template<> const int IO<3, 1>::counter = 16;
template<> const int IO<3, 2>::counter = 20;
template<> const int IO<3, 3>::counter = 16;
template<> const int IO<3, 4>::counter = 12;
template<> const int IO<4, 0>::counter = 9;
template<> const int IO<4, 1>::counter = 12;
template<> const int IO<4, 2>::counter = 15;
template<> const int IO<4, 3>::counter = 12;
template<> const int IO<4, 4>::counter = 9;



template<int ytype, int xtype>
class TPS
{
public:
	static const float stencil[5][5];

	template <class CNEI>
	float get(int index, CNEI &nei, float* src_data)
	{
		float sum = 0.0f;
		if (stencil[0][0] != 0.0f)	sum += src_data[nei.ll(nei.tt(index))] * stencil[0][0];
		if (stencil[0][1] != 0.0f)	sum += src_data[nei.l(nei.tt(index))] * stencil[0][1];
		if (stencil[0][2] != 0.0f)	sum += src_data[nei.tt(index)] * stencil[0][2];
		if (stencil[0][3] != 0.0f)	sum += src_data[nei.r(nei.tt(index))] * stencil[0][3];
		if (stencil[0][4] != 0.0f)	sum += src_data[nei.rr(nei.tt(index))] * stencil[0][4];
		if (stencil[1][0] != 0.0f)	sum += src_data[nei.ll(nei.t(index))] * stencil[1][0];
		if (stencil[1][1] != 0.0f)	sum += src_data[nei.l(nei.t(index))] * stencil[1][1];
		if (stencil[1][2] != 0.0f)	sum += src_data[nei.t(index)] * stencil[1][2];
		if (stencil[1][3] != 0.0f)	sum += src_data[nei.r(nei.t(index))] * stencil[1][3];
		if (stencil[1][4] != 0.0f)	sum += src_data[nei.rr(nei.t(index))] * stencil[1][4];
		if (stencil[2][0] != 0.0f)	sum += src_data[nei.ll(index)] * stencil[2][0];
		if (stencil[2][1] != 0.0f)	sum += src_data[nei.l(index)] * stencil[2][1];
		if (stencil[2][2] != 0.0f)	sum += src_data[index] * stencil[2][2];
		if (stencil[2][3] != 0.0f)	sum += src_data[nei.r(index)] * stencil[2][3];
		if (stencil[2][4] != 0.0f)	sum += src_data[nei.rr(index)] * stencil[2][4];
		if (stencil[3][0] != 0.0f)	sum += src_data[nei.ll(nei.b(index))] * stencil[3][0];
		if (stencil[3][1] != 0.0f)	sum += src_data[nei.l(nei.b(index))] * stencil[3][1];
		if (stencil[3][2] != 0.0f)	sum += src_data[nei.b(index)] * stencil[3][2];
		if (stencil[3][3] != 0.0f)	sum += src_data[nei.r(nei.b(index))] * stencil[3][3];
		if (stencil[3][4] != 0.0f)	sum += src_data[nei.rr(nei.b(index))] * stencil[3][4];
		if (stencil[4][0] != 0.0f)	sum += src_data[nei.ll(nei.bb(index))] * stencil[4][0];
		if (stencil[4][1] != 0.0f)	sum += src_data[nei.l(nei.bb(index))] * stencil[4][1];
		if (stencil[4][2] != 0.0f)	sum += src_data[nei.bb(index)] * stencil[4][2];
		if (stencil[4][3] != 0.0f)	sum += src_data[nei.r(nei.bb(index))] * stencil[4][3];
		if (stencil[4][4] != 0.0f)	sum += src_data[nei.rr(nei.bb(index))] * stencil[4][4];

		return sum;
	}

	template <class CNEI>
	void update(int index, CNEI &nei, float d, float* dst_data)
	{
		if (stencil[0][0] != 0.0f)	dst_data[nei.ll(nei.tt(index))] += stencil[0][0] * d;
		if (stencil[0][1] != 0.0f)	dst_data[nei.l(nei.tt(index))] += stencil[0][1] * d;
		if (stencil[0][2] != 0.0f)	dst_data[nei.tt(index)] += stencil[0][2] * d;
		if (stencil[0][3] != 0.0f)	dst_data[nei.r(nei.tt(index))] += stencil[0][3] * d;
		if (stencil[0][4] != 0.0f)	dst_data[nei.rr(nei.tt(index))] += stencil[0][4] * d;
		if (stencil[1][0] != 0.0f)	dst_data[nei.ll(nei.t(index))] += stencil[1][0] * d;
		if (stencil[1][1] != 0.0f)	dst_data[nei.l(nei.t(index))] += stencil[1][1] * d;
		if (stencil[1][2] != 0.0f)	dst_data[nei.t(index)] += stencil[1][2] * d;
		if (stencil[1][3] != 0.0f)	dst_data[nei.r(nei.t(index))] += stencil[1][3] * d;
		if (stencil[1][4] != 0.0f)	dst_data[nei.rr(nei.t(index))] += stencil[1][4] * d;
		if (stencil[2][0] != 0.0f)	dst_data[nei.ll(index)] += stencil[2][0] * d;
		if (stencil[2][1] != 0.0f)	dst_data[nei.l(index)] += stencil[2][1] * d;
		if (stencil[2][2] != 0.0f)	dst_data[index] += stencil[2][2] * d;
		if (stencil[2][3] != 0.0f)	dst_data[nei.r(index)] += stencil[2][3] * d;
		if (stencil[2][4] != 0.0f)	dst_data[nei.rr(index)] += stencil[2][4] * d;
		if (stencil[3][0] != 0.0f)	dst_data[nei.ll(nei.b(index))] += stencil[3][0] * d;
		if (stencil[3][1] != 0.0f)	dst_data[nei.l(nei.b(index))] += stencil[3][1] * d;
		if (stencil[3][2] != 0.0f)	dst_data[nei.b(index)] += stencil[3][2] * d;
		if (stencil[3][3] != 0.0f)	dst_data[nei.r(nei.b(index))] += stencil[3][3] * d;
		if (stencil[3][4] != 0.0f)	dst_data[nei.rr(nei.b(index))] += stencil[3][4] * d;
		if (stencil[4][0] != 0.0f)	dst_data[nei.ll(nei.bb(index))] += stencil[4][0] * d;
		if (stencil[4][1] != 0.0f)	dst_data[nei.l(nei.bb(index))] += stencil[4][1] * d;
		if (stencil[4][2] != 0.0f)	dst_data[nei.bb(index)] += stencil[4][2] * d;
		if (stencil[4][3] != 0.0f)	dst_data[nei.r(nei.bb(index))] += stencil[4][3] * d;
		if (stencil[4][4] != 0.0f)	dst_data[nei.rr(nei.bb(index))] += stencil[4][4] * d;
	}
};

template<> const float TPS<0, 0>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 8, -8, 2 }, { 0, 0, -8, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<0, 1>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, -8, 20, -12, 2 }, { 0, 4, -12, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<0, 2>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 2, -12, 22, -12, 2 }, { 0, 4, -12, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<0, 3>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 2, -12, 20, -8, 0 }, { 0, 4, -12, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<0, 4>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 2, -8, 8, 0, 0 }, { 0, 4, -8, 0, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<1, 0>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, -8, 4, 0 }, { 0, 0, 20, -12, 2 }, { 0, 0, -12, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<1, 1>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 4, -12, 4, 0 }, { 0, -12, 36, -16, 2 }, { 0, 4, -16, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<1, 2>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 4, -12, 4, 0 }, { 2, -16, 38, -16, 2 }, { 0, 4, -16, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<1, 3>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 4, -12, 4, 0 }, { 2, -16, 36, -12, 0 }, { 0, 4, -16, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<1, 4>::stencil[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 4, -8, 0, 0 }, { 2, -12, 20, 0, 0 }, { 0, 4, -12, 0, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<2, 0>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 0, -12, 4, 0 }, { 0, 0, 22, -12, 2 }, { 0, 0, -12, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<2, 1>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -16, 4, 0 }, { 0, -12, 38, -16, 2 }, { 0, 4, -16, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<2, 2>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -16, 4, 0 }, { 2, -16, 40, -16, 2 }, { 0, 4, -16, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<2, 3>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -16, 4, 0 }, { 2, -16, 38, -12, 0 }, { 0, 4, -16, 4, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<2, 4>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -12, 0, 0 }, { 2, -12, 22, 0, 0 }, { 0, 4, -12, 0, 0 }, { 0, 0, 2, 0, 0 } };
template<> const float TPS<3, 0>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 0, -12, 4, 0 }, { 0, 0, 20, -12, 2 }, { 0, 0, -8, 4, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<3, 1>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -16, 4, 0 }, { 0, -12, 36, -16, 2 }, { 0, 4, -12, 4, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<3, 2>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -16, 4, 0 }, { 2, -16, 38, -16, 2 }, { 0, 4, -12, 4, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<3, 3>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -16, 4, 0 }, { 2, -16, 36, -12, 0 }, { 0, 4, -12, 4, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<3, 4>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -12, 0, 0 }, { 2, -12, 20, 0, 0 }, { 0, 4, -8, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<4, 0>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 0, -8, 4, 0 }, { 0, 0, 8, -8, 2 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<4, 1>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -12, 4, 0 }, { 0, -8, 20, -12, 2 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<4, 2>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -12, 4, 0 }, { 2, -12, 22, -12, 2 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<4, 3>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -12, 4, 0 }, { 2, -12, 20, -8, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };
template<> const float TPS<4, 4>::stencil[5][5] = { { 0, 0, 2, 0, 0 }, { 0, 4, -8, 0, 0 }, { 2, -8, 8, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };



template<int ytype, int xtype>
class Oper
{
public:
	IO<ytype, xtype> io;
	TPS<ytype, xtype> tps;
};
