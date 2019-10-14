#include <iostream>

#include "cuda_common.cuh"

#define VERBOSE
#ifndef ENABLE_CUDA
#define ENABLE_CUDA
#endif

namespace CUDA{
	bool g_CUDA_f_is_initialized = false;

	int g_CUDA_block_size_1D = 0;

	int g_CUDA_block_size_2DX = 0;
	int g_CUDA_block_size_2DY = 0;
	
	int g_CUDA_block_size_3DX = 0;
	int g_CUDA_block_size_3DY = 0;
	int g_CUDA_block_size_3DZ = 0;

	bool*  gpu_mask = NULL;
	float* gpu_z_plane = NULL;
	bool*  gpu_boundary = NULL;
	float* gpu_result_x = NULL;
	float* gpu_result_y = NULL;
  float* gpu_result = NULL;
  float* energy_each = NULL;

	int g_CUDA_max_shared_mem_size = 0;

  void GPUinit(const bool* mask, const float* z_plane, const bool* boundary, const int rows, const int cols, bool multiple_devices){

		if(g_CUDA_f_is_initialized)
			return;

		int dev_number = 0;

		if(multiple_devices){
			int count;

			CUDA_SAFE_CALL( cudaGetDeviceCount(&count) );

			size_t max_mem = 0;
			int best_dvc_number;

			for (int iter_dvc = 0; iter_dvc<count; ++iter_dvc){
				int 					 devID_temp;
				cudaDeviceProp props_temp;

				CUDA_SAFE_CALL( cudaSetDevice(iter_dvc) );
				CUDA_SAFE_CALL( cudaGetDevice(&devID_temp) );

				if (iter_dvc != devID_temp)
					throw Exception( "Error initializeGPU(): Could not set device %d to get memory properties\n", iter_dvc);

				CUDA_SAFE_CALL( cudaGetDeviceProperties(&props_temp, devID_temp) );

				if (props_temp.totalGlobalMem > max_mem){
					max_mem					= props_temp.totalGlobalMem;
					best_dvc_number = iter_dvc;
				}
			}

			dev_number = best_dvc_number;
		}

		CUDA_SAFE_CALL (cudaSetDevice(dev_number));

		int								devID;
		cudaDeviceProp 		props;

		CUDA_SAFE_CALL( cudaGetDevice(&devID) );
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, devID) );

#ifdef VERBOSE
		printf( "\n---------------------- Initializing GPU ----------------------\n" );
		printf( "Device detected: #%d \"%s\" with Compute %d.%d capability\n",
				devID, props.name,
				props.major, props.minor );
		printf( "totalGlobalMem: %.2f MB\n",     (float)props.totalGlobalMem/1024.0f/1024.0f );
		printf( "sharedMemPerBlock: %.2f KB\n",  (float)props.sharedMemPerBlock/1024.0f );
		printf( "registersPerBlock: %d\n",       props.regsPerBlock );
		printf( "warpSize: %d\n",                props.warpSize );
		printf( "maxThreadsPerBlock: %d\n",      props.maxThreadsPerBlock );
		printf( "maxThreadsDim: %d x %d x %d\n", props.maxThreadsDim[0],
				props.maxThreadsDim[1],
				props.maxThreadsDim[2] );
		printf( "maxGridSize: %d x %d x %d\n",   props.maxGridSize[0],
				props.maxGridSize[1],
				props.maxGridSize[2] );
		printf( "clockRate: %.2f MHz\n",         (float)props.clockRate/1000.0f );
		printf( "multiProcessorCount: %d\n",     props.multiProcessorCount );
		printf( "canMapHostMemory: %d\n",        props.canMapHostMemory );
#endif

		size_t memGPUFree = 0;
		size_t memGPUTotal = 0;
		CUDA_SAFE_CALL( cudaMemGetInfo( &memGPUFree, &memGPUTotal ) );

#ifdef VERBOSE
		printf( "cudaMemGetInfo: Free/Total[MB]: %.1f/%.1f\n",
				(float)memGPUFree/1048576, (float)memGPUTotal/1048576 );
#endif

		bool f_ok = true;
		if ( props.major < 2)
			f_ok = false;
		else if ( (props.major == 2) && (props.minor <0 ) )
			f_ok = false;

		if ( !f_ok){
			throw Exception( "Error initializeGPU(): Need CUDA computing "
					"capabilities > 2.0\nDetected version ist %d.%d\n"
					"Disable CUDA-usage for enabling the CPU algorithms\n",
					props.major, props.minor );
		}

		// CUDA allowing mapping host memory?
		if ( !props.canMapHostMemory )
		{
			throw Exception( "Error initializeGPU(): Device can't map host memory\n"
					"Disable CUDA-usage for enabling the CPU algorithms\n" );
		}

		// CUDA allowing 3D threads-blocks?
		if ( props.maxThreadsDim[2] <= 1 )
		{
			throw Exception( "Error initializeGPU(): Device does not allow 3D "
					"thread-blocks:\nmaxThreadsDim: %d x %d x %d\n"
					"Disable CUDA-usage for enabling the CPU algorithms\n",
					props.maxThreadsDim[0], props.maxThreadsDim[1],
					props.maxThreadsDim[2] );
		}

		// CUDA allowing 3D grid-size?
		if ( props.maxGridSize[2] <= 1 )
		{
			throw Exception( "Error initializeGPU(): Device does not allow 3D "
					"grid-size:\nmaxGridSize: %d x %d x %d\n"
					"Disable CUDA-usage for enabling the CPU algorithms\n",
					props.maxGridSize[0], props.maxGridSize[1],
					props.maxGridSize[2] );
		}

		if (props.maxThreadsPerBlock >= 1024){
			g_CUDA_block_size_1D = 512;

			g_CUDA_block_size_2DX = 32;
			g_CUDA_block_size_2DY = 16;

			g_CUDA_block_size_3DX = 16;
			g_CUDA_block_size_3DY = 8;
			g_CUDA_block_size_3DZ = 8;
		}else if (props.maxThreadsPerBlock >= 512){
			g_CUDA_block_size_1D = 256;

			g_CUDA_block_size_2DX = 16;
			g_CUDA_block_size_2DY = 16;

			g_CUDA_block_size_3DX = 8;
			g_CUDA_block_size_3DY = 8;
			g_CUDA_block_size_3DZ = 4;
		}else{
			int nx1D = (int) 1 < (int)(props.maxThreadsPerBlock) ? (int)(props.maxThreadsPerBlock) : (int) 1;

			int nx2D = (int) 1 < (int)(props.maxThreadsPerBlock/8) ? (int)(props.maxThreadsPerBlock/8) : (int) 1;
			int ny2D = props.maxThreadsPerBlock / nx2D;

			int nx3D = (int) 1 < (int)(props.maxThreadsPerBlock/4) ? (int)(props.maxThreadsPerBlock/4) : (int) 1;
			int nLeft = props.maxThreadsPerBlock / nx3D;
			int ny3D = (int) 1 < (int)(nLeft/4) ? (int)(nLeft/4) : (int) 1;
			int nz3D = nLeft / ny3D;

			g_CUDA_block_size_1D = nx1D;

			g_CUDA_block_size_2DX = nx2D;
			g_CUDA_block_size_2DY = ny2D;
			
			g_CUDA_block_size_3DX = nx3D;
			g_CUDA_block_size_3DY = ny3D;
			g_CUDA_block_size_3DZ = nz3D;
		}

		g_CUDA_max_shared_mem_size = props.sharedMemPerBlock;

#ifdef VERBOSE
		printf( "blockSize1D = %d ; blockSize2D = %dx%d ; blockSize3D = %dx%dx%d\n",
				g_CUDA_block_size_1D, g_CUDA_block_size_2DX, g_CUDA_block_size_2DY,
				g_CUDA_block_size_3DX, g_CUDA_block_size_3DY, g_CUDA_block_size_3DZ );
		printf( "--------------------------------------------------------------\n\n" );
#endif

		g_CUDA_f_is_initialized = true;

		CUDA_SAFE_CALL( cudaMalloc( (void**) &gpu_mask, rows*cols*sizeof(bool) ) );
		CUDA_SAFE_CALL( cudaMemcpy(gpu_mask, mask, rows*cols*sizeof(bool), cudaMemcpyHostToDevice) );

		CUDA_SAFE_CALL( cudaMalloc( (void**) &gpu_z_plane, rows*cols*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMemcpy(gpu_z_plane, z_plane, rows*cols*sizeof(float), cudaMemcpyHostToDevice) );

		CUDA_SAFE_CALL( cudaMalloc( (void**) &gpu_boundary, rows*cols*sizeof(bool) ) );
		CUDA_SAFE_CALL( cudaMemcpy(gpu_boundary, boundary, rows*cols*sizeof(bool), cudaMemcpyHostToDevice) );

		CUDA_SAFE_CALL( cudaMalloc( (void**) &gpu_result_x, rows*cols*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**) &gpu_result_y, rows*cols*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**) &gpu_result, rows*cols*sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**) &energy_each, rows*cols*sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMemset( (void*)  energy_each, 0, rows*cols*sizeof(float)) );
	}

	void GPUclose(){
		CUDA_SAFE_CALL( cudaFree(gpu_mask) );
		CUDA_SAFE_CALL( cudaFree(gpu_z_plane) );
		CUDA_SAFE_CALL( cudaFree(gpu_boundary) );
		CUDA_SAFE_CALL( cudaFree(gpu_result_x) );
		CUDA_SAFE_CALL( cudaFree(gpu_result_y) );
    CUDA_SAFE_CALL( cudaFree(gpu_result) );
    CUDA_SAFE_CALL( cudaFree(energy_each) );
    //cudaDeviceReset(); // Not necessary and very costly
		g_CUDA_f_is_initialized = false;
	}

	void get2DGridBlock(int width, int height, dim3 &dimGrid, dim3 &dimBlock){
		dimBlock = dim3 (g_CUDA_block_size_2DX, g_CUDA_block_size_2DY, 1);

		int grid_size_X = (width + dimBlock.x-1) / dimBlock.x;
		int grid_size_Y = (height + dimBlock.y-1) / dimBlock.y;

		dimGrid = dim3( grid_size_X, grid_size_Y, 1);
	}
}
