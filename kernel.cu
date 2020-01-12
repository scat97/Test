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
    if(pos_x >= width*height) return;

    int val = gray[pos_x];
    //int k = (pos_x * 4)%width;
    //res[pos_x] = static_cast<unsigned char>( 255 * (val-min)/(max-min));
    //int row = pos_x/width;
    //int x = pos_x-row*width;
    //res[pos_x] = (255);


    if(val > max) res[pos_x] =255;
    else if(val < min) res[pos_x] =0;
    else res[pos_x] = static_cast<unsigned char> (255 * (val-min)/(max-min));
    
    //uchar newval = 50;// static_cast<unsigned char>  (255 * (val-min)/(max-min));
    //surf2Dwrite(newval, surftest, x*4, row);
    
    /*if(val > max) {
      unsigned char maxval = 255;
      surf2Dwrite(maxval, surftest, x, row);
    }
    else if (val < min) {
      unsigned char minval = 0;
      surf2Dwrite(minval, surftest, x, row);}

    else {
      unsigned char newval = static_cast<unsigned char>  (255 * (val-min)/(max-min));
      surf2Dwrite(newval, surftest, x, row);
    }*/



}

__global__ void Smoothing(unsigned char*gray,unsigned char*res,int width, int height){
  int pos_x = blockIdx.x*blockDim.x+threadIdx.x;
  int pos_y = blockIdx.y*blockDim.y+threadIdx.y;
  
  if ((pos_x >= width*height) || (pos_y >= height)) return;

  /*if ((pos_x == 0) && (pos_y == 0)) {
    res[0] = static_cast<unsigned char>((gray[0] + gray[1] +gray[width] + gray[width+1]) / 9.0); }
     // top left
  else if ((pos_x == (width-1))&&(pos_y == 0)){
    res[width-1] = static_cast<unsigned char>((gray[width-1] + gray[width-2] +  gray[width*2-1] + gray[width*2-2]) / 9.0) ; }
    //top right
  else if ((pos_x == (width-1)) && (pos_y == (height-1))){
     res[width*height-1] = static_cast<unsigned char>( (gray[height*width-1] + gray[height*width-2] + gray[(height-1)*width-1] + gray[(height-1)*width-2])/9.0) ; }//4bottom right}

  else if ((pos_x == 0) && (pos_y == (height-1))) {
    res[(height-1)*width] =  static_cast<unsigned char>((gray[(height-1)*width] + gray[(height-1)*width + 1] + gray[(height-2)*width] + gray[(height-2)*width+1]) / 9.0); }//4bottom left}

  else if(pos_y == 0){
    res[pos_x] =static_cast<unsigned char>( (gray[pos_x] + gray[pos_x+1] + gray[pos_x-1] + gray[ width+ pos_x-1] + gray[ width+ pos_x] + gray[ width+ pos_x+1]) / 9.0); // 6top row
  }
  else if(pos_y == (height-1)){
     res[pos_x+(height-1)*width] = static_cast<unsigned char>( (gray[(height-1)*width+pos_x] + gray[(height-1)*width+pos_x+1] + gray[(height-1)*width+pos_x-1] + gray[(height-2)*width+ pos_x-1] + gray[(height-2)*width+ pos_x] + gray[ (height-2)*width+ pos_x+1]) / 9.0);} // top row
  else if(pos_x == 0){
    res[width*pos_y] = static_cast<unsigned char>((gray[width*pos_y] + gray[width*pos_y+1] + gray[width*(pos_y-1)] + gray[width*(pos_y-1)+1] + gray[width*(pos_y+1)] + gray[width*(pos_y+1)+1]) / 9.0);}
  else if(pos_x == (width-1)){
    res[width*(pos_y+1)-1] = static_cast<unsigned char>((gray[width*(pos_y+1) - 1] + gray[width*(pos_y+1) - 1 -1] + gray[width*(pos_y) - 1] + gray[width*(pos_y) - 1 -1] + gray[width*(pos_y+2) - 1] +  gray[width*(pos_y+2) - 1 - 1]) / 9.0);}
  else {
    unsigned char val = static_cast<unsigned char>(((gray[pos_y * width + pos_x] + gray[pos_y * width + pos_x -1 ] + gray[pos_y * width + pos_x +1 ] + gray[(pos_y - 1)* width + pos_x  ] + gray[(pos_y + 1)* width + pos_x  ]  + gray[(pos_y-1) * width + pos_x-1] + gray[(pos_y-1) * width + pos_x+1] + gray[(pos_y+1) * width + pos_x-1] + gray[(pos_y+1) * width + pos_x+1] ) / 9.0 ));
    res[width*pos_y+pos_x] = val;}
*/
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

__global__ void Smoothing_new(unsigned char*gray,unsigned char *res,int width, int height, const int blwd, const int blht){

  int width2 = blockDim.x;
  int pos_x2 = threadIdx.x;
  int pos_y2 = threadIdx.y;
  __shared__ pixel tempout[18*18];

 tempout[pos_x2+width2*pos_y2].tleft = 0;
  tempout[pos_x2+width2*pos_y2].tmid = 0;
  tempout[pos_x2+width2*pos_y2].tright = 0;
  tempout[pos_x2+width2*pos_y2].mleft = 0;
  tempout[pos_x2+width2*pos_y2].mmid = 0;
  tempout[pos_x2+width2*pos_y2].mright = 0;
  tempout[pos_x2+width2*pos_y2].bleft = 0;
  tempout[pos_x2+width2*pos_y2].bmid = 0;
  tempout[pos_x2+width2*pos_y2].bright = 0;
  int pos_x = blockIdx.x*(blockDim.x-2)+threadIdx.x;
  int pos_y = blockIdx.y*(blockDim.y-2)+threadIdx.y;
  
  if ((pos_x >= width) || (pos_y >= height)) return;


    int val = static_cast<int>(gray[pos_x+pos_y*width]);
    
    tempout[pos_x2+width2*pos_y2].bright+=val;
    tempout[pos_x2+1+width2*pos_y2].bmid+=val;
    tempout[pos_x2+2+width2*pos_y2].bleft+=val;

    tempout[pos_x2+width2*(pos_y2+1)].mright+=val;
    tempout[pos_x2+1+width2*(pos_y2+1)].mmid+=val;
    tempout[pos_x2+2+width2*(pos_y2+1)].mleft+=val;

    tempout[pos_x2+width2*(pos_y2+2)].tright+=val;
    tempout[pos_x2+1+width2*(pos_y2+2)].tmid+=val;
    tempout[pos_x2+2+width2*(pos_y2+2)].tleft+=val;
    //

  if((threadIdx.x == 0) || (threadIdx.x == blockDim.x-1) || (threadIdx.y == 0) || (threadIdx.y == blockDim.y-1)){
    return;
  }
  //if((pos_x < 20) && (pos_y ==1)){printf("%d %d\n", pos_x, tempout[pos_x2+width2*pos_y2].mid );}
  __syncthreads();
  val = tempout[pos_x2+1+width2*(pos_y2+1)].tleft+tempout[pos_x2+1+width2*(pos_y2+1)].tmid+tempout[pos_x2+1+width2*(pos_y2+1)].tright+tempout[pos_x2+1+width2*(pos_y2+1)].mleft+tempout[pos_x2+1+width2*(pos_y2+1)].mmid+tempout[pos_x2+1+width2*(pos_y2+1)].mright+tempout[pos_x2+1+width2*(pos_y2+1)].bleft+tempout[pos_x2+1+width2*(pos_y2+1)].bmid+tempout[pos_x2+1+width2*(pos_y2+1)].bright;

  //res[pos_x+width*pos_y] = static_cast<unsigned char> ((tempout[pos_x2+1+width2*(pos_y2+1)].left+tempout[pos_x2+1+width2*(pos_y2+1)].mid+tempout[pos_x2+1+width2*(pos_y2+1)].right)/9.0);

  if((pos_x == 0)){
    if((pos_y == 0) || (pos_y == width-1)){
      res[pos_x+width*pos_y] = static_cast<unsigned char> (val/4.0);
    }
    else 
      res[pos_x+width*pos_y] = static_cast<unsigned char> (val/6.0);
  }
  else if(pos_x == width-1){
    if(pos_y == 0){
      res[pos_x+width*pos_y] = static_cast<unsigned char> (val/4.0);
    }
    else if(pos_y == width-1){
      res[pos_x+width*pos_y] = static_cast<unsigned char> (val/4.0);
    }
    else 
      res[pos_x+width*pos_y] = static_cast<unsigned char> (val/6.0);
  }
  else if(pos_y == 0){
    res[pos_x+width*pos_y] = static_cast<unsigned char> (val/6.0);
  }
  else if(pos_y == height-1){
    res[pos_x+width*pos_y] = static_cast<unsigned char> (val/6.0);
  }
  else res[pos_x+width*pos_y] = static_cast<unsigned char> (val/9.0);
  
  //unsigned char data1 = static_cast<unsigned char> (tex2D(texRef,pos_x,pos_y)*255);
  /*unsigned char data1;
  int mul = 4;
  surf2Dread(&data1,surftest, pos_x*mul, pos_y);
  
  unsigned char data2;
  unsigned char data3;
  unsigned char data4;
  unsigned char data5;
  unsigned char data6;
  unsigned char data7;
  unsigned char data8;
  unsigned char data9;
  int mul = 1;
  surf2Dread(&data1,surftest, pos_x*mul, pos_y);

  surf2Dread(&data2,surftest, (pos_x+1)*mul , (pos_y-1));
  surf2Dread(&data3,surftest, (pos_x-1)*mul, pos_y-1);
  surf2Dread(&data4,surftest, pos_x*mul , pos_y-1);
  surf2Dread(&data5,surftest, (pos_x-1)*mul, pos_y);
  surf2Dread(&data6,surftest, (pos_x+1)*mul, pos_y);
  surf2Dread(&data7,surftest, (pos_x)*mul, pos_y+1);
  surf2Dread(&data8,surftest, (pos_x+1)*mul, pos_y+1);
  surf2Dread(&data9,surftest, (pos_x-1)*mul, pos_y+1);*/
//res[pos_x+width*pos_y] = tex2D(texRef,pos_x,pos_y;//static_cast<unsigned char>((data1+data2+data3+data4+data5+data6+data7+data8+data9)/9.0);
  /*
  __shared__ int interresult[blockDim.x+2][blockDim.y+2];

  for(int i = -1; i < 2; i++){
    atomicAdd(&interresult[pos_x+i+1][(pos_y)],gray[pos_x+(pos_y)*width]);
    atomicAdd(&interresult[pos_x+i+1][(pos_y+2)],gray[pos_x+(pos_y)*width]);
    atomicAdd(&interresult[pos_x+i+1][(pos_y+1)],gray[pos_x+(pos_y)*width]);
  }

  __syncthreads();
  res[(pos_x+1)+(width+2)*(pos_y+1)] = interresult[pos_x+1][pos_y+1];
  if((threadIdx.x == 0) | (threadIdx.x = blockDim.x-1) | (threadIdx.y == 0) | (threadIdx.y == blockDim.y-1)){
    atomicAdd(&res[pos_x])
  }
*/
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
