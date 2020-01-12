#include <cuda_runtime.h>
//#define cimg_use_jpeg
#include "CImg.h"
#include <iostream>
#include "kernel.h"
#include "kernelcpu.h"
//#include "helper_cuda.h"
#define BLOCKSIZE 128
#define BLOCKSIZE2 16
using namespace std;
using namespace cimg_library;


/////////////////
//surface<void, 2> surftest;

//////////

int compute_diff(unsigned char * res_cpu, unsigned char * res_gpu, unsigned long size){
  int res = 0;
  for(int i = 0;i < size; i++){
    res += res_cpu[i] - res_gpu[i];
  }
  return res;
}

int compute_diff_hist(int * res_cpu, int * res_gpu, unsigned long size){
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
    unsigned char *d_dst_2;
    //cudaMalloc(&d_dst_2, width*height*sizeof(unsigned int));

    unsigned char *GPU_contrast;
    unsigned char *GPU_smoothing;
    unsigned char *GPU_smoothing2;

    cudaEvent_t start; // to record processing time
    cudaEvent_t stop;
    float msecTotal,msecTotal2;
  


    std::cout << "Start CPU processing" << std::endl;
    // create and start timer

    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    unsigned char *cpu_ref = new unsigned char [width*height];
    int* cpu_ref_hist = new int[256];
    memset(cpu_ref_hist, 0, sizeof(int)*256);

    rgb2gray_cpu(h_src, cpu_ref, width, height);

    // hisgram cpu
    
    histgram_cpu(cpu_ref_hist, cpu_ref, width,  height);
    // stop and destroy timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    float cpu_time = msecTotal;

    std::cout <<"CPU processing time: " << cpu_time << " ms" <<std::endl;

    
    //std::cout << "Start GPU processing" << std::endl;
    // create and start timer
    //cudaEventCreate(&start);
    //cudaEventRecord(start, NULL); 

    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_dst, width*height*sizeof(unsigned char));

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

////////////////////////////////////////////////
/// RGB to gray ///

    //launch the kernel
    dim3 blkDim (BLOCKSIZE, 1,1);
    dim3 grdDim (ceil(size2/BLOCKSIZE), 1, 1);
    cout << (ceil(size2/BLOCKSIZE)) << " grid" << endl;
    // create and start timer
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 

    rgb2gray<<<grdDim, blkDim>>>(d_src, d_dst, width, height);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    cout << "RGB2Gray time GPU:" << msecTotal << "ms" << endl;

    cudaDeviceSynchronize();
    cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);
    
//////////////////////////////////////////////////
    //for(int AmountOfHists = 16; AmountOfHists <= 16; AmountOfHists+= 4){
        int power = 2;
        int AmountOfHists = pow(2,power);
        int* hist = new int[256];
        int* histGPU = new int[256*AmountOfHists];
        // create and start timer
    ///////////////////////////////////////////////////////
        dim3 blkDim2 (256, 1,1);
        dim3 grdDim2 (ceil(size2/256),1, 1);
        cudaEventCreate(&start);
        cudaEventRecord(start, NULL); 

        cudaMalloc(&histGPU, 256*sizeof(int)*AmountOfHists);
        cudaMemset(histGPU, 0, 256*sizeof(int)*AmountOfHists);
        //cout << "malloc is " << temp3 << " memset is " << temp2 <<endl;
        int mask = pow(2,(power))-1;

        histgram<<<grdDim2,blkDim2>>>(histGPU, d_dst, width , mask);

        dim3 blkDim3 (256,1,1);
        for (int stride = AmountOfHists/2; stride>0; stride>>=1){ // Interleaved reduction
            dim3 grdDim3 (stride,1,1);
            histgram_summation<<<grdDim3, blkDim3>>>(histGPU, stride);
            cudaDeviceSynchronize();
        }
  
        cudaMemcpy(hist, histGPU, 256*sizeof(int),cudaMemcpyDeviceToHost);
        
        cudaEventCreate(&stop);
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msecTotal, start, stop);
        int diff_hist;
        diff_hist = compute_diff_hist(hist, cpu_ref_hist, 256);
        if(diff_hist == 0){
            cout << "Histogram time GPU:" << msecTotal << "ms" << endl;
        }

        cudaFree(histGPU);
    //}
    ///////////////////////////////////////////////////////

    //int* hist = new int[256]();
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
   
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    cudaError_t error3 = cudaMalloc((void**)&GPU_contrast, width*height*sizeof(unsigned char));

    if (error3 != cudaSuccess){
        cout << "Error 3" << endl;
        if(error3 == cudaErrorInvalidValue){
            cout<< "Invalid value" << endl;
        }
        if(error3 == cudaErrorInvalidSurface){
            cout<< "Invalid surface" << endl;
        }

    }

    dim3 grdBlkCon (128, 1,1);
    dim3 grdDimCon (ceil(size2/128),1,1);
    ContrastEnhancement<<<grdDimCon,grdBlkCon>>>(d_dst,GPU_contrast,width,height,min,max);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal2, start, stop);
    cout << "Contrast enhancement time GPU:" << msecTotal2 << "ms" << endl;






    //cout << +h_contrast[(height-1)*width]<< " " << +h_contrast[(height-1)*width+1] << " " << +h_contrast[(height-2)*width] << " "  << +h_contrast[(height-2)*width+1] << endl;
   /* cudaMalloc(&d_dst_2, (width*height)*sizeof(unsigned char));

    dim3 blkDimSmth (16, 16, 1);
    dim3 grdDimSmth ((width + 16-1)/BLOCKSIZE2, (height + BLOCKSIZE2-1)/BLOCKSIZE2, 1);
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    Smoothing<<<grdDimSmth,blkDimSmth>>>(GPU_contrast,d_dst_2, width, height);
    
    // add other three kernels here
    // clock starts -> copy data to gpu -> kernel1 -> kernel2->kernel3->kernel 4 ->copy result to cpu -> clock stops

    //wait until kernel finishes
    cudaDeviceSynchronize();
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    //cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);
    
    //copy back the result to CPU
    //cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);
    unsigned char* temp5 = new unsigned char [width*height];
    
    cudaMemcpy(temp5, d_dst_2, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    std::cout << "correct: " << +temp5[12132] << " " << +temp5[120] << " " << +temp5[8294400/2] << " " << +temp5[8294400/2+1] << endl;
*/
    cout << "Time of original smoothing function: " << msecTotal << endl;
    cudaMalloc(&GPU_smoothing2, width*height*sizeof(unsigned char));




    //cout << +h_contrast[(height-1)*width]<< " " << +h_contrast[(height-1)*width+1] << " " << +h_contrast[(height-2)*width] << " "  << +h_contrast[(height-2)*width+1] << endl;

    int xthreads = 16;
    int ythreads = 16;
    cudaDeviceSynchronize();
    dim3 blkDimSmthnew (xthreads, ythreads, 1);
    dim3 grdDimSmthnew (ceil(width/(xthreads-2)), ceil(height/(ythreads-2)), 1);
    //int maxdim = (xthreads+2)*(ythreads+2);
    cudaEventCreate(&start);
    cudaEventRecord(start, NULL); 
    Smoothing_new<<<grdDimSmthnew,blkDimSmthnew>>>(GPU_contrast,GPU_smoothing2, width, height, xthreads, ythreads);
    cudaDeviceSynchronize();
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    cout << "Time of optimised smoothing function: " << msecTotal << endl;

    if ( cudaSuccess != cudaGetLastError() )
    cout << "Error!\n";
    //wait until kernel finishes


    unsigned char* temp2 = new unsigned char[width*height];
    
    error3 = cudaMemcpy(h_dst, GPU_smoothing2, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (error3 != cudaSuccess){
        cout << "Error 3" << endl;
        if(error3 == cudaErrorInvalidValue){
            cout<< "Invalid value" << endl;
        }
        if(error3 == cudaErrorInvalidSurface){
            cout<< "Invalid surface" << endl;
        }

    }

    //cout << "new: " << +temp2[1] << " " << +temp2[0] <<" " << +temp2[8294400/2] << " " << +temp2[8294400/2+1] << endl;

    
/*cout << "size " << size2 << endl;
int i=0;
    while(i < size2){
        if(temp2[i] != 0){
            cout << i << " integer " << +temp2[i] << endl;
            break;
        }
        i++;
    }*/
    //int res = compute_diff(h_dst,temp5,width*height);
    
  //  cout << res << " diff " << endl;


    cudaFree(GPU_contrast);
    cudaFree(GPU_smoothing);
    cudaFree(GPU_smoothing2);
    //cudaFree(histGPU);
    cudaFree(d_src);


    cudaFree(d_src);
    cudaFree(d_dst);

    cudaDeviceReset();
   // std::cout << "diff cpu and gpu " << res <<std::endl; // do not change this
    //std::cout <<"CPU processing time: " << gpu_time << " ms" <<std::endl; // do not change this
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
    //cout << +h_dst[height*width]<<endl;
    return 0;
}
