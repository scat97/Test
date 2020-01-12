#include <cuda_runtime.h>
#include <iostream>
#include <iostream>
//surface<void, cudaSurfaceType2D> surftest;
typedef unsigned char uchar;
//texture<uchar, cudaTextureType2D, cudaReadModeElementType> texRef;
struct pixel{
    uchar tleft;
    uchar tmid;
    uchar tright;
    uchar mleft;
    uchar mmid;
    uchar mright;
    uchar bleft;
    uchar bmid;
    uchar bright;
};
// declear you kernels
__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height);

void rgb2gray_cpu(unsigned char * d_src, unsigned char * d_dst, int width, int height);
void histgram_cpu(int* hist, unsigned char*gray,int width, int height);

__global__ void histgram(int* hist, unsigned char*gray,int width, int height);
__global__ void histgram_summation(int* hist, int num_of_hists);

__global__ void ContrastEnhancement(unsigned char*gray,unsigned char*res,int width, int height, int min, int max);

__global__ void Smoothing(unsigned char*gray,unsigned char*res,int width, int height);
__global__ void Smoothing_new(unsigned char*gray,unsigned char *res,int width, int height, const int blwd,const int blht);

