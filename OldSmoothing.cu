  __shared__ int tempout[34+width2*34];

  tempout[pos_x2][pos_y2] = 0;
  int pos_x = blockIdx.x*(blockDim.x-2)+threadIdx.x;
  int pos_y = blockIdx.y*(blockDim.y-2)+threadIdx.y;
  
  if ((pos_x >= width) || (pos_y >= height)) return;


    int val = static_cast<int>(gray[pos_x+pos_y*width]);
    
    tempout[pos_x2][pos_y2]+=val;
    __syncthreads();
    tempout[pos_x2+1][pos_y2]+=val;
    __syncthreads();
    tempout[pos_x2+2][pos_y2]+=val;
    __syncthreads();
    tempout[pos_x2][pos_y2+1]+=val;
    __syncthreads();
    tempout[pos_x2+1][pos_y2+1]+=val;
    __syncthreads();
    tempout[pos_x2+2][pos_y2+1]+=val;
    __syncthreads();
    tempout[pos_x2][pos_y2+2]+=val;
    __syncthreads();
    tempout[pos_x2+1][pos_y2+2]+=val;
    __syncthreads();
    tempout[pos_x2+2][pos_y2+2]+=val;
    __syncthreads();


    __global__ void Smoothing_new(unsigned char*gray,unsigned char *res,int width, int height, int maxdim){
  int width2 = blockDim.x;
  int pos_x2 = threadIdx.x;
  int pos_y2 = threadIdx.y;
  __shared__ pixel tempout[130*10];

 /* tempout[pos_x2+width2*pos_y2].left = 0;
  tempout[pos_x2+width2*pos_y2].mid = 0;
  tempout[pos_x2+width2*pos_y2].right = 0;*/
  int pos_x = blockIdx.x*(blockDim.x-2)+threadIdx.x;
  int pos_y = blockIdx.y*(blockDim.y-2)+threadIdx.y;
  
  if ((pos_x >= width) || (pos_y >= height)) return;


    int val = static_cast<int>(gray[pos_x+pos_y*width]);
    
    tempout[pos_x2+width2*pos_y2].right+=val;
    tempout[pos_x2+1+width2*pos_y2].mid+=val;
    tempout[pos_x2+2+width2*pos_y2].left+=val;

    __syncthreads();
    tempout[pos_x2+width2*(pos_y2+1)].right+=val;
    tempout[pos_x2+1+width2*(pos_y2+1)].mid+=val;
    tempout[pos_x2+2+width2*(pos_y2+1)].left+=val;

    __syncthreads();
    tempout[pos_x2+width2*(pos_y2+2)].right+=val;
    tempout[pos_x2+1+width2*(pos_y2+2)].mid+=val;
    tempout[pos_x2+2+width2*(pos_y2+2)].left+=val;
    //

  if((threadIdx.x == 0) || (threadIdx.x == blockDim.x-1) || (threadIdx.y == 0) || (threadIdx.y == blockDim.y-1)){
    return;
  }
  //if((pos_x < 20) && (pos_y ==1)){printf("%d %d\n", pos_x, tempout[pos_x2+width2*pos_y2].mid );}
  __syncthreads();
  res[pos_x+width*pos_y] = static_cast<unsigned char> ((tempout[pos_x2+1+width2*(pos_y2+1)].left+tempout[pos_x2+1+width2*(pos_y2+1)].mid+tempout[pos_x2+1+width2*(pos_y2+1)].right)/9.0);
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

 
const int tw = blockDim.x;
const int th = blockDim.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int pos_x = blockIdx.x*(blockDim.x)+threadIdx.x;
int pos_y = blockIdx.y*(blockDim.y)+threadIdx.y;
__shared__ int tempin[18][18];

if(tx == 0){ // left column
  tempin[ty][tx+1] = 0;
}
else if(tx == tw-1){ // right column
  tempin[ty+2][tx+1] = 0;
}

if(ty == 0){
  tempin[ty+1][tx+1] = 0; // upper part
  if(tx == 0){
  tempin[ty][0] = 0; }// top left corner}
  else if (tx == tw-1){
  tempin[ty][tw] = 0; // top right corner
  }
}
else if(ty == th-1){
  tempin[ty+1][tx+1] = 0; // upper part
  if(tx == 0){
  tempin[ty][0] = 0; }// top left corner}
  else if (tx == tw-1){
  tempin[ty][tw] = 0; // top right corner
  }
}

tempin[ty+1][tx+1] = gray[pos_x+width*pos_y];

if((tx == 0) && (pos_x >0)) tempin[ty+1][tx] = gray[pos_x-1+(pos_y)*width]; // left row // potential ERROR!!!
if((tx == tw-1) && (pos_x < width-1)) tempin[ty+1][tx+2] = gray[pos_x+1+(pos_y)*width]; // right row
if((ty == 0) && (pos_y > 0)) { 
  tempin[ty][tx+1] = gray[pos_x+(pos_y-1)*width]; // toprow
  if((tx == 0) && (pos_x > 0)) tempin[ty][tx] = gray[pos_x-1+(pos_y-1)*width]; //top left corner
  else if((tx == tw-1) && (pos_x < width-1)) tempin[th][tw] = gray[pos_x+1+(pos_y-1)*width];// top right corner
}
if((ty == th-1) && (pos_y < height-1)){
  tempin[ty+2][tx+1] = gray[pos_x+(pos_y+1)*width];
  if((tx == 0) && (pos_x > 0)) tempin[ty][tx] = gray[pos_x-1+(pos_y+1)*width]; //bottom left corner
  else if((tx == tw-1) && (pos_x < width-1)) tempin[th][tw] = gray[pos_x+1+(pos_y+1)*width];// bottom right corner
}

__syncthreads();
if ((pos_x >= width) || (pos_y >= height)) return;

int val = tempin[ty][tx+1] + tempin[ty][tx] + tempin[ty][tx+2]+tempin[ty+1][tx+1] + tempin[ty+1][tx] + tempin[ty+1][tx+2]+tempin[ty+2][tx+1] + tempin[ty+2][tx] + tempin[ty+2][tx+2];
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
