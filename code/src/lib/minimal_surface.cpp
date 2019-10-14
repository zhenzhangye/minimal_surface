#include "minimal_surface.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#include "cuda/cuda_common.cuh" 
#include "cuda/cuda_kernels.cuh" 
MinimalSurface::MinimalSurface(const int rows, const int cols, const int max_iter, float tol, bool verbose):
                rows_(rows), cols_(cols), max_iter_(max_iter), tol_(tol), verbose_(verbose),
							 	mask_(NULL), z_plane_(NULL), boundary_(NULL), area_(0), plane_volume_(0){
}

MinimalSurface::~MinimalSurface(){
	CUDA::GPUclose();
	FreeMemorySpace();
}

bool MinimalSurface::FreeMemorySpace(){
	if(mask_!=NULL){
		delete [] mask_;
		mask_ = NULL;
	}

	if(z_plane_!=NULL){
		delete [] z_plane_;
		z_plane_ = NULL;
	}

	if(boundary_!=NULL){
		delete [] boundary_;
		boundary_ = NULL;
	}

}


bool MinimalSurface::AllocateMemorySpace(const bool *mask, const float *z_plane, const bool *boundary){
	mask_ = new bool[rows_*cols_];
	std::copy(mask, mask + rows_*cols_ , mask_);

	z_plane_ = new float[rows_*cols_];
	std::copy(z_plane, z_plane + rows_*cols_, z_plane_);

	boundary_ = new bool[rows_*cols_];
	std::copy(boundary, boundary + rows_*cols_, boundary_);

	for(int i = 0; i<rows_*cols_; ++i){
		if(mask_[i] && !boundary_[i])
			++ area_;
	}

	for(int i = 0; i<rows_*cols_; ++i){
		if(mask_[i]){
			plane_volume_ += z_plane_[i];
		}
	}
	CUDA::GPUinit(mask, z_plane, boundary, rows_, cols_);
	
	return true;
}

void MinimalSurface::Ballooning(const float *z, const float tau, const float target_volume, float* result){
	float* result_x = new float[rows_*cols_];
	float* result_y = new float[rows_*cols_];
  CUDA::ComputeGradient(result, rows_, cols_, tau, target_volume, plane_volume_, area_, max_iter_, tol_, verbose_);
	/*
	for(size_t i = 0; i<cols_; ++i){
		for(size_t j = 0; j<rows_; ++j){
			std::cout<<result[i*rows_+j]<<" ";
		}
		std::cout<<std::endl;
	}
	*/
	delete [] result_x;
	delete [] result_y;
}

void MinimalSurface::Ballooning(const float *z, const float tau, const float target_volume, float* result, const float q, const float tolerance){
	float* result_x = new float[rows_*cols_];
	float* result_y = new float[rows_*cols_];
	//CUDA::FISTA(mask_, z, result_x, result_y, result, rows_, cols_, tau, target_volume, z_plane_, boundary_, plane_volume_, area_, max_iter_, q, tolerance);
	std::cout<<tolerance<<std::endl;
	/*
	for(size_t i = 0; i<cols_; ++i){
		for(size_t j = 0; j<rows_; ++j){
			std::cout<<result[i*rows_+j]<<" ";
		}
		std::cout<<std::endl;
	}
	*/
	delete [] result_x;
	delete [] result_y;
}

void MinimalSurface::PrintInformation(){
	std::cout<<"-----------------mask--------------------------"<<std::endl;
	for(int i = 0; i < cols_; ++i){
		for(int j = 0; j < rows_; ++j){
			std::cout<<mask_[i*rows_+j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<"----------------boundary-----------------------"<<std::endl;
	for(int i = 0; i < cols_; ++i){
		for(int j = 0; j< rows_; ++j){
			std::cout<<boundary_[i*rows_+j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<"---------------z_plane-------------------------"<<std::endl;
	for(int i = 0; i < cols_; ++i){
		for(int j = 0; j< rows_; ++j){
			std::cout<<z_plane_[i*rows_+j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<"area: "<<area_<<" plane_volume: "<<plane_volume_<<std::endl;
}
