#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"
// implement your kernels
__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    //int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    /*if (pos_x >= width || pos_y >= height)
        return;
*/
    /*
     * CImg RGB channels are split, not interleaved.
     * (http://cimg.eu/reference/group__cimg__storage.html)
     */
    unsigned char r = d_src[pos_x];
    unsigned char g = d_src[height* width + pos_x];
    unsigned char b = d_src[height * 2 * width + pos_x];

    d_dst[pos_x] = (unsigned char)((float)(r + g + b) / 3.0f + 0.5);
    //unsigned char gray = _gray > 255 ? 255 : _gray;
}

__global__ void histgram(int* hist, unsigned char * gray, int width, int height){
    __shared__ int histshared[256];
    if(threadIdx.x < 256){
    histshared[threadIdx.x] = 0;
    }
    __syncthreads();
    int pos_x = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned char loc = gray[pos_x];
    atomicAdd(&histshared[loc], 1);
    __syncthreads();
    if (threadIdx.x >255) return;
    if(histshared[threadIdx.x] == 0) return;
    
    atomicAdd(&hist[threadIdx.x+(blockIdx.x&height)*256], histshared[threadIdx.x]);
    


}
__global__ void histgram_summation(int* hist, int stride){
  //int pos_x = blockIdx.x*blockDim.x+threadIdx.x;
  //int pos_y = blockIdx.y*blockDim.y+threadIdx.y;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  hist[bid*256+tid]+=hist[(stride+bid)*256+tid];
}

__global__ void ContrastEnhancement(unsigned char*gray,unsigned char*res,int width, int height, int min, int max){
    int pos_x = blockIdx.x*blockDim.x+threadIdx.x;
    int pos_y = blockIdx.y*blockDim.y+threadIdx.y;

    if (pos_x >= width || pos_y >= height)
    return;

    int val = gray[pos_y*width+pos_x];
    if(val > max){
      res[pos_y*width+pos_x] = 255;
    }
    else if(val < min){
      res[pos_y*width+pos_x] = 0;
    }
    else{
      res[pos_y*width+pos_x] = static_cast<unsigned char>( 255 * (val-min)/(max-min));
    }


}

__global__ void Smoothing(unsigned char*gray,unsigned char*res,int width, int height){
  int pos_x = blockIdx.x*blockDim.x+threadIdx.x;
  int pos_y = blockIdx.y*blockDim.y+threadIdx.y;
  
  if (pos_x >= width || pos_y >= height) return;

  if ((pos_x == 0) && (pos_y == 0)) {
    res[0] = static_cast<unsigned char>((gray[0] + gray[1] +gray[width] + gray[width+1]) / 4.0); }
     // top left
  else if ((pos_x == (width-1))&&(pos_y == 0)){
    res[width-1] = static_cast<unsigned char>((gray[width-1] + gray[width-2] +  gray[width*2-1] + gray[width*2-2]) / 4.0) ; }
    //top right
  else if ((pos_x == (width-1)) && (pos_y == (height-1))){
     res[width*height-1] = static_cast<unsigned char>( (gray[height*width-1] + gray[height*width-2] + gray[(height-1)*width-1] + gray[(height-1)*width-2])/4.0) ; }//bottom right}

  else if ((pos_x == 0) && (pos_y == (height-1))) {
    res[(height-1)*width] =  static_cast<unsigned char>((gray[(height-1)*width] + gray[(height-1)*width + 1] + gray[(height-2)*width] + gray[(height-2)*width+1]) / 4.0); }//bottom left}

  else if(pos_y == 0){
    res[pos_x] =static_cast<unsigned char>( (gray[pos_x] + gray[pos_x+1] + gray[pos_x-1] + gray[ width+ pos_x-1] + gray[ width+ pos_x] + gray[ width+ pos_x+1]) / 6.0); // top row
  }
  else if(pos_y == (height-1)){
     res[pos_x+(height-1)*width] = static_cast<unsigned char>( (gray[(height-1)*width+pos_x] + gray[(height-1)*width+pos_x+1] + gray[(height-1)*width+pos_x-1] + gray[(height-2)*width+ pos_x-1] + gray[(height-2)*width+ pos_x] + gray[ (height-2)*width+ pos_x+1]) / 6.0);} // top row
  else if(pos_x == 0){
    res[width*pos_y] = static_cast<unsigned char>((gray[width*pos_y] + gray[width*pos_y+1] + gray[width*(pos_y-1)] + gray[width*(pos_y-1)+1] + gray[width*(pos_y+1)] + gray[width*(pos_y+1)+1]) / 6.0);}
  else if(pos_x == (width-1)){
    res[width*(pos_y+1)-1] = static_cast<unsigned char>((gray[width*(pos_y+1) - 1] + gray[width*(pos_y+1) - 1 -1] + gray[width*(pos_y) - 1] + gray[width*(pos_y) - 1 -1] + gray[width*(pos_y+2) - 1] +  gray[width*(pos_y+2) - 1 - 1]) / 6.0);}
  else {
    unsigned char val = static_cast<unsigned char>(((gray[pos_y * width + pos_x] + gray[pos_y * width + pos_x -1 ] + gray[pos_y * width + pos_x +1 ] + gray[(pos_y - 1)* width + pos_x  ] + gray[(pos_y + 1)* width + pos_x  ]  + gray[(pos_y-1) * width + pos_x-1] + gray[(pos_y-1) * width + pos_x+1] + gray[(pos_y+1) * width + pos_x-1] + gray[(pos_y+1) * width + pos_x+1] ) / 9.0 ));
    res[width*pos_y+pos_x] = val;}
}

////////////////////////////////////////// CPU functions
void rgb2gray_cpu(unsigned char * d_src, unsigned char * d_dst, int width, int height){

  for (int i = 0; i < width ; i++){
    for (int j = 0; j < height ; j++){
      unsigned char r = d_src[j * width + i];
      unsigned char g = d_src[(height + j ) * width + i];
      unsigned char b = d_src[(height * 2 + j) * width + i];
      unsigned int _gray = (unsigned int)((float)(r + g + b) / 3.0f + 0.5);
      unsigned char gray = _gray > 255 ? 255 : _gray;
      d_dst[j * width + i] = gray;
    }
  }

}

void histgram_cpu(int* hist, unsigned char*gray,int width, int height){
  int size = width*height;

  for (int i=0;i<size;i++){
      unsigned char gray_val=gray[i];
      hist[gray_val]++;
  }
}
