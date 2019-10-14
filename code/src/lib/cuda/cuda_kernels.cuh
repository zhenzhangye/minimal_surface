#ifndef MINIMAL_SURFACE_CUDA_KERNELS_CUH_
#define MINIMAL_SURFACE_CUDA_KERNELS_CUH_
#include <stddef.h>

namespace CUDA{
  void ComputeGradient(float* result,
                       const int rows,
                       const int cols,
                       const float tau,
                       const float target_volume,
                       const float plane_volume,
                       const int area,
                       const int max_iter,
                       float tol,
                       bool verbose);

}
#endif
