#ifndef MINIMAL_SURFACE_H_
#define MINIMAL_SURFACE_H_
#include <iostream>

class MinimalSurface{
public:
  MinimalSurface(const int rows, const int cols, const int max_iter, float tol, bool verbose);
	~MinimalSurface();

	bool AllocateMemorySpace(const bool *mask, const float *z_plane, const bool *boundary);
	bool FreeMemorySpace();
	void Ballooning(const float* z, const float tau, const float target_volume, float *result);
	void Ballooning(const float* z, const float tau, const float target_volume, float *result, const float q, const float tolerance);

	void PrintInformation();

	int  getNumberRows() {return rows_;};
	int  getNumberCols() {return cols_;};

private:
  bool 		        verbose_;
  bool* 					mask_;
  bool*						boundary_;
	float*					z_plane_;
	int 						rows_;
	int 						cols_;
  int							max_iter_;
  float					  tol_;
  int							area_;
	float						plane_volume_;
};

#endif
