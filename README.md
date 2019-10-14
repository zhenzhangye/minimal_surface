# minimal_surface
This code implements the approach for [this paper](https://vision.in.tum.de/_media/spezial/bib/windheuser-et-al-miru12.pdf) with gradient descent on GPU:

> **Fast and Globally Optimal Single View Reconstruction of Curved Objects**
> *M. R. Oswald, E. Toeppe and D. Cremers; In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.*
![alt tag](https://vision.in.tum.de/_media/spezial/bib/oswald_toeppe_cremers_cvpr12.jpg)

Based on a minimal user input, this algorithm interactively determines the objects' silhouette and subsequently computes a silhouette-consistent 3D model which is precisely the globally minimal surface with user-specified volume.

## 1. Requirements

This code has two three party dependencies:

0) MATLAB (mandatory)

1) [CUDA](https://developer.nvidia.com/cuda-zone) (mandatory)

2) [CMake](https://cmake.org/) (mandatory)

## 2. Getting started
This code is tested under:
* Ubuntu 16.04
* CMake 3.5.1
* CUDA 8.0
* MATLAB R2019a

Download the demo [dataset](https://vision.in.tum.de/data/datasets/photometricdepthsr):
* `cd pathto/code/data`

* `./download.sh`

Build Mex file:
* Set `MATLAB_ROOT` environment variable in to your installed matlab path, such as

  `export MATLAB_ROOT='/usr/local/MATLAB/R2019a'` in `~/.bashrc`

* In Terminal do
  `cd pathto/code`

  `mkdir build`

  `cmake ..`

  `make`

* In `build/lib` is the mex file and in `build/bin` is the binary

* `cd ..`

* execute the matlab script `example.m`

## 3. Input
- `mask`: a boolean binary describing the silhouette.
- `volume`: the desired volume of result.

## 4. Parameters
- `max_iter`: the max number of iteration for gradient descent.
- `tol`: the stopping criterion by residual.
- `tau`: the step size for gradient descent.

## 5. License

minimal_surface is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, see [here](http://creativecommons.org/licenses/by-nc-sa/4.0/), with an additional request:

If you make use of the library in any form in a scientific publication, please refer to `https://github.com/zhenzhangye/minimal_surface` and cite the paper

```
@inProceedings{Oswald-et-al-cvpr12,
  author    = {M. R. Oswald and E. Toeppe and D. Cremers},
  title     = {Fast and Globally Optimal Single View Reconstruction of Curved Objects},
  booktitle = cvpr,
  year      = {2012},
  address   = {Providence, Rhode Island},
  month     = jun,
  titleurl  = {oswald_toeppe_cremers_cvpr12.pdf},
  topic = {Single View Reconstruction},
  pages = {534-541},
  keywords = {convex-relaxation, singleview}
}

```
