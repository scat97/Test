#include <cuda_runtime.h>
//#define cimg_use_jpeg
#include "CImg.h"
#include <iostream>
#include "kernel.h"
#include "kernelcpu.h"
#define BLOCKSIZE 32
using namespace std;
using namespace cimg_library;


int compute_diff(unsigned char * res_cpu, unsigned char * res_gpu, unsigned long size){
  int res = 0;
  for(int i = 0;i < size; i++){
    res += res_cpu[i] - res_gpu[i];
  }
  return res;
}

int main()
{
    //load image
    CImg<unsigned char> src("cat2.jpg"); // we use cat2.jpg to grade
    int width = src.width();
    int height = src.height();
    unsigned long size = src.size();
    unsigned long size2 = width*height;

    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, dev);

    //create pointer to image
    unsigned char *h_src = src.data();
    
    CImg<unsigned char> dst(width, height, 1, 1);
    unsigned char *h_dst = dst.data();
    //Something contrast blabla
    CImg<unsigned char> contrast(width, height, 1, 1);
    unsigned char *h_contrast = contrast.data();

    // for contrast enhancemant
    CImg<unsigned char> smoothing_gpu(width, height, 1, 1);
    unsigned char *h_smoothing = smoothing_gpu.data();

    unsigned char *d_src;
    unsigned char *d_dst;

    unsigned char *GPU_contrast;
    unsigned char *GPU_smoothing;

    cudaEvent_t start; // to record processing time
    cudaEvent_t stop;
    float msecTotal;
  


    std::cout << "Start CPU processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    // hisgram cpu
    unsigned char *cpu_ref = new unsigned char [width*height];
    rgb2gray_cpu(h_src,cpu_ref,width,height);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time = msecTotal;

    std::cout <<"CPU processing time: " << cpu_time << " ms" <<std::endl;

    
    std::cout << "Start GPU processing" << std::endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_dst, width*height*sizeof(unsigned char));

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    //launch the kernel
    dim3 blkDim (BLOCKSIZE, BLOCKSIZE, 1);
    dim3 grdDim ((width + BLOCKSIZE-1)/BLOCKSIZE, (height + BLOCKSIZE-1)/BLOCKSIZE, 1);
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    rgb2gray<<<grdDim, blkDim>>>(d_src, d_dst, width, height);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    cout << "RGB2Gray time GPU:" << msecTotal << endl;

    int* hist = new int[256];
    int* histGPU = new int[256];
    cudaMalloc(&histGPU, 256*sizeof(int));
    cudaMemset(histGPU, 0, 256*sizeof(int));
    histgram<<<grdDim,blkDim>>>(histGPU, d_dst, width , height);
    cudaDeviceSynchronize();
    cudaMemcpy(hist, histGPU, 256*sizeof(int),cudaMemcpyDeviceToHost);
    int min,max;
    int temp = 0;

    for (int i = 0;i<255;i++){
        temp += hist[i];
        if(temp > size2*0.1){
            min = i;
            temp = 0;
            break;
        }
    }
    temp = 0;
    for (int i=255;i>=0;i--){
        temp += hist[i];
        if(temp > size2*0.1){
            max = i;
            temp = 0;
            break;
        }
    }
    cout << "min " << min << " max " << max << endl;
    cudaMalloc((void**)&GPU_contrast, width*height*sizeof(unsigned char));
    ContrastEnhancement<<<grdDim,blkDim>>>(d_dst,GPU_contrast,width,height,min,max);

    cudaMemcpy(h_contrast,GPU_contrast, width*height, cudaMemcpyDeviceToHost);
    cout << +h_contrast[(height-1)*width]<< " " << +h_contrast[(height-1)*width+1] << " " << +h_contrast[(height-2)*width] << " "  << +h_contrast[(height-2)*width+1] << endl;
    cudaMalloc(&GPU_smoothing, width*height*sizeof(unsigned char));
    Smoothing<<<grdDim,blkDim>>>(GPU_contrast,d_dst, width, height);
    cout << "Managed" << endl;
    
    // add other three kernels here
    // clock starts -> copy data to gpu -> kernel1 -> kernel2->kernel3->kernel 4 ->copy result to cpu -> clock stops

    //wait until kernel finishes
    cudaDeviceSynchronize();
    cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);
    
    //copy back the result to CPU
    //cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float gpu_time = msecTotal;
    

    
    int res = compute_diff(cpu_ref,h_dst,width*height);

    cudaFree(GPU_contrast);
    cudaFree(GPU_smoothing);
    cudaFree(histGPU);
    cudaFree(d_src);


    cudaFree(d_src);
    cudaFree(d_dst);

    cudaDeviceReset();
    std::cout << "diff cpu and gpu " << res <<std::endl; // do not change this
    std::cout <<"CPU processing time: " << gpu_time << " ms" <<std::endl; // do not change this
    //you need to save your final output, we need to measure the correctness of your program
    //read test.cpp to learn how to save a image
    //smoothing_gpu.save("smoothing_gpu.jpg"); 
  
    FILE * pFile;
    pFile = fopen ("gpu_out.txt","w");
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            fprintf(pFile, "%d ", +h_dst[j*width+i]);
        }
        fprintf(pFile, "\n");
    }
    fclose(pFile);
    contrast.save("Con_GPU.jpg");
    //&h_dst = h_src;
    dst.save("file.jpg");
    cout << +h_dst[height*width]<<endl;
    return 0;
}
