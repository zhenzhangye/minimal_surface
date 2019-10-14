#include <iostream>
#include <cmath>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "cuda_common.cuh"
#include "cuda_kernels.cuh"

namespace CUDA{


	__device__ void ComputeNormalizedGradient(float &x, float &y, float grad_x, float grad_y){
    x = grad_x / sqrtf(grad_x * grad_x + grad_y * grad_y + 1);
    y = grad_y / sqrtf(grad_x * grad_x + grad_y * grad_y + 1);
	}
  __global__ void ComputeGradientKernel(const bool *mask, const float *depth, float *result_x, float *result_y, const int rows, const int cols){
		int x = threadIdx.x + blockDim.x * blockIdx.x;
		int y = threadIdx.y + blockDim.y * blockIdx.y;

		if (x<rows && y<cols){



      float grad_x;
      float grad_y;
      int ind = x + rows * y;

      int x_neighbor = x + rows * y + 1;
      int y_neighbor = x + rows * (y+1);

      // compute the gradient on x direction.
      if (!mask[ind]){
        grad_x = 0;
      }else if (x+1 >= rows){ // apply Dirichlet boundary condition
        grad_x = -depth[ind];
      }else if (!mask[x_neighbor]){ // apply Dirichlet boundary condition
        grad_x = -depth[ind];
      }else{
        grad_x = -depth[ind] + depth[x_neighbor];
      }

      // compute the gradient on y direction.
      if (!mask[ind]){
        grad_y = 0;
      }else if (y+1 >= cols){
        grad_y = -depth[ind];
      }else if (!mask[y_neighbor]){
        grad_y = -depth[ind];
      }else{
        grad_y = -depth[ind] + depth[y_neighbor];
      }

      ComputeNormalizedGradient(result_x[ind], result_y[ind], grad_x, grad_y);
    }

	}

	__global__ void ComputeEnergyKernel(const float *depth, const bool *mask, const int rows, const int cols, float *energy_each){
		int x = threadIdx.x + blockDim.x * blockIdx.x;
		int y = threadIdx.y + blockDim.y * blockIdx.y;

		if (x<rows && y<cols){
			float grad_x;
			float grad_y;
			int ind = x + rows * y;
			int x_neighbor = x + rows * y + 1;
			int y_neighbor = x + rows * (y+1);

			// compute the gradient on x direction.
			if (!mask[ind]){
				grad_x = 0;
			}else if (x+1 >= rows){
				grad_x = -depth[ind];
			}else if (!mask[x_neighbor]){
				grad_x = -depth[ind];
			}else{
				grad_x = -depth[ind] + depth[x_neighbor];
			}

			// compute the gradient on y direction.
			if (!mask[ind]){
				grad_y = 0;
			}else if (y+1 >= cols){
				grad_y = -depth[ind];
			}else if (!mask[y_neighbor]){
				grad_y = -depth[ind];
			}else{
				grad_y = -depth[ind] + depth[y_neighbor];
			}

      energy_each[ind] = sqrtf(1 + grad_x*grad_x + grad_y*grad_y);
		}
	}

	__global__ void PerformGradientStepKernel(const bool *mask, const float *result_x, const float *result_y, float *result, const int rows, const int cols, const float tau, const float *z_plane, const bool *boundary){
		int x = threadIdx.x + blockDim.x * blockIdx.x;
		int y = threadIdx.y + blockDim.y * blockIdx.y;

		if (x<rows && y<cols){
			int ind = x + rows * y;
			int x_neighbor = x + rows * y - 1;
			int y_neighbor = x + rows * (y-1);

			// compute dEdz
			if (!mask[ind]){
				result[ind] = 0;
			}
			else{
				if (boundary[ind]){
          result[ind] = z_plane[ind];
				}else{

          float value = -result_x[ind] - result_y[ind];

					if (x_neighbor>=0){
						if(mask[x_neighbor]){
							value += result_x[x_neighbor];
						}
					}
					if (y_neighbor>=0){
						if(mask[y_neighbor]){
							value += result_y[y_neighbor];
						}
					}
					result[ind] = result[ind] - tau*value;
				}

			}
		}
	}

	__global__ void UpdateDepthKernel(const bool* mask, const bool* boundary, float *result, const int rows, const int cols, const float remaining_volume){
		int x = threadIdx.x + blockDim.x * blockIdx.x;
		int y = threadIdx.y + blockDim.y * blockIdx.y;

		if (x<rows && y<cols){
      int ind = x + rows * y;
      if( mask[ind] && !boundary[ind] ){
        result[ind] -= remaining_volume;
      }
		}
	}

  void ComputeGradient(float* result,
                       const int rows,
                       const int cols,
                       const float tau,
                       const float target_volume,
                       const float plane_volume,
                       const int area,
                       const int max_iter,
                       float tol,
                       bool verbose){
		dim3 dimGrid, dimBlock;
		get2DGridBlock(rows, cols, dimGrid, dimBlock);

    float old_energy = 0.f;
    for (size_t i = 0; i<max_iter; ++i){
			ComputeGradientKernel<<< dimGrid, dimBlock >>> (gpu_mask, gpu_result, gpu_result_x, gpu_result_y, rows, cols);
			CUDA_SAFE_CALL( cudaDeviceSynchronize() );

			PerformGradientStepKernel<<< dimGrid, dimBlock >>> (gpu_mask, gpu_result_x, gpu_result_y, gpu_result, rows, cols, tau, gpu_z_plane, gpu_boundary);
			CUDA_SAFE_CALL( cudaDeviceSynchronize() );

			float current_volume = thrust::reduce(thrust::device, gpu_result, gpu_result + rows*cols);
			float remaining_volume = (target_volume - (plane_volume-current_volume)) / area;

			UpdateDepthKernel<<< dimGrid, dimBlock >>> (gpu_mask, gpu_boundary, gpu_result, rows, cols, remaining_volume);
			CUDA_SAFE_CALL( cudaDeviceSynchronize() );

      ComputeEnergyKernel<<< dimGrid, dimBlock >>> (gpu_result, gpu_mask, rows, cols, energy_each);

      if (i%100==0){
        float current_energy = thrust::reduce(thrust::device, energy_each, energy_each + rows*cols);
        if (verbose)
          std::cout<< "E[" << i << "]: " << current_energy << " - Residual: " << fabs((old_energy-current_energy)/current_energy) << std::endl;
        if (old_energy>current_energy && fabs((old_energy-current_energy)/current_energy)<tol)
          break;
        else
          old_energy = current_energy;
      }
		}

		CUDA_SAFE_CALL( cudaMemcpy(result, gpu_result, rows*cols*sizeof(float), cudaMemcpyDeviceToHost) );

	}

}
