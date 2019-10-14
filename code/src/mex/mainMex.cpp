#include <string>
#include <functional>
#include <map>
#include <sstream>

#include "../lib/minimal_surface.h"

// matlab headers
#include "mex.h"

// project dependency
#include "mex_utils.h"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

static void initMinimalSurface(MEX_ARGS){
	MinimalSurface *minimal_surface;

	if(!mxIsEmpty(prhs[0])){
		bool *mask = mxGetLogicals(prhs[0]);
		const mwSize *dims = mxGetDimensions(prhs[0]);
		int rows = dims[0];
		int cols = dims[1];

		float *z_plane = (float*)mxGetPr(prhs[1]);
    bool  *boundary = (bool*)mxGetPr(prhs[2]);
    int max_iter = (int)mxGetScalar(prhs[3]);
    float tol = (float)mxGetScalar(prhs[4]);
    bool verbose = (float)mxGetScalar(prhs[5]);
    minimal_surface = new MinimalSurface(rows, cols, max_iter, tol, verbose);
		if(!minimal_surface->AllocateMemorySpace(mask, z_plane, boundary)){
			delete minimal_surface;
			return;
		}
	}
	plhs[0] = ptr_to_handle(minimal_surface);
}

static void GradientDescent(MEX_ARGS){
	MinimalSurface *minimal_surface = handle_to_ptr<MinimalSurface>(prhs[0]);
	float *z = (float*)mxGetPr(prhs[1]);
	float tau = (float)mxGetScalar(prhs[2]);
	float target_volume = (float)mxGetScalar(prhs[3]);

	const mwSize *dims = mxGetDimensions(prhs[1]);
	plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
	float *result = (float*)mxGetPr(plhs[0]);
	minimal_surface->Ballooning(z, tau, target_volume, result);
}

static void FISTA(MEX_ARGS){
	MinimalSurface *minimal_surface = handle_to_ptr<MinimalSurface>(prhs[0]);
	float *z = (float*)mxGetPr(prhs[1]);
	float tau = (float)mxGetScalar(prhs[2]);
	float target_volume = (float)mxGetScalar(prhs[3]);
	float tolerance = (float)mxGetScalar(prhs[4]);
	float momentum = (float)mxGetScalar(prhs[5]);

	const mwSize *dims = mxGetDimensions(prhs[1]);
	plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
	float *result = (float*)mxGetPr(plhs[0]);

	minimal_surface->Ballooning(z, tau, target_volume, result, momentum, tolerance);
}

static void closeMinimalSurface(MEX_ARGS){
	MinimalSurface *minimal_surface = handle_to_ptr<MinimalSurface>(prhs[0]);
	delete minimal_surface;
}

const static std::map<std::string, std::function<void(MEX_ARGS)>> cmd_reg = {
	{ "initMinimalSurface", 	initMinimalSurface },
	{ "GradientDescent",			GradientDescent },
	{ "closeMinimalSurface", 	closeMinimalSurface },
	{ "FISTA",								FISTA },
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  if(nrhs == 0)
		mexErrMsgTxt("Usage: MinimalSurfaceMEX(command, arg1, arg2, ...);");

	char *cmd = mxArrayToString(prhs[0]);
	bool executed = false;

	for(auto& c : cmd_reg){
		if(c.first.compare(cmd) == 0){
			c.second(nlhs, plhs, nrhs-1, prhs+1);
			executed = true;
			break;
		}
	}

	if(!executed){
		std::stringstream msg;
		msg << "Unkown comman '" << cmd << "'. List of supported commands:";
		for (auto& c : cmd_reg)
			msg << "\n - " << c.first.c_str();

		mexErrMsgTxt(msg.str().c_str());
	}

	mexEvalString("drawnow;");

	return;
}
