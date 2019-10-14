# minimal_surface
This code implements the approach for [this paper](https://vision.in.tum.de/_media/spezial/bib/windheuser-et-al-miru12.pdf) with gradient descent on GPU:

> **Fast and Globally Optimal Single View Reconstruction of Curved Objects**
> *M. R. Oswald, E. Toeppe and D. Cremers; In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.*
![alt tag](https://vision.in.tum.de/_media/spezial/bib/oswald_toeppe_cremers_cvpr12.jpg)

Based on aminimal user input, this algorithm interactively determines the objects' silhouette and subsequently computes a silhouette-consistent 3D model which is precisely the globally minimal surface with user-specified volume.

## 1. Requirements

This code has two three party dependencies:

0) MATLAB (works under MATLAB R2019a)

1) [CUDA](https://developer.nvidia.com/cuda-zone) (works under CUDA-8.0)

## 2. Input
- One RGB image `I`.
- A blooean binary `mask` describing the silhouette in the RGB image.
