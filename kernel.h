#include <cuda_runtime.h>
#include <iostream>

// declear you kernels
__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height);

void rgb2gray_cpu(unsigned char * d_src, unsigned char * d_dst, int width, int height);

__global__ void histgram(int* hist, unsigned char*gray,int width, int height);

__global__ void ContrastEnhancement(unsigned char*gray,unsigned char*res,int width, int height, int min, int max);

__global__ void Smoothing(unsigned char*gray,unsigned char*res,int width, int height);

