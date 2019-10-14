#ifndef MINIMAL_SURFACE_CUDA_COMMON_CUH_
#define MINIMAL_SURFACE_CUDA_COOMON_CUH_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../exception.h"

#ifdef MAX
#undef MAX
#endif
#ifdef MIN
#undef MIN
#endif

#define CUDA_SAFE_CALL(x) { gpuAssert((x), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
}

namespace CUDA{
	extern bool g_CUDA_f_is_initialized;
	
	extern int g_CUDA_block_size_1D;

	extern int g_CUDA_block_size_2DX;
	extern int g_CUDA_block_size_2DY;

	extern int g_CUDA_block_size_3DX;
	extern int g_CUDA_block_size_3DY;
	extern int g_CUDA_block_size_3DZ;

	extern int g_CUDA_max_shared_mem_size;
	
	extern bool* 	gpu_mask;
	extern bool* 	gpu_boundary;
	extern float* gpu_z_plane;
	extern float* gpu_result_x;
	extern float* gpu_result_y;
	extern float* gpu_result;
  extern float* energy_each;

	void get2DGridBlock(int width, int height, dim3 &dim_grid, dim3 &dim_block);

	void GPUinit(const bool* mask, const float* z_plane, const bool* boundary, const int rows, const int cols, bool multiple_devices = false);

	void GPUclose( void );
}

#endif

