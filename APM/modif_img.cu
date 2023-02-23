#include <iostream>
#include <string.h>
#include "FreeImage.h"
#include <stdio.h>
#include "cuda.h"
#include <cuda_runtime.h>

#define THRESHOLD 50
#define WIDTH 3840
#define HEIGHT 2160
#define BLOCK_WIDTH 32
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

__global__
void kernel_saturation(unsigned int* c_d_img,unsigned int* c_d_tmp, int height, int width){
   
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;
  int id;
  id = ((line * width) + col)*3;

  if ((line < height -1) && (col < width -1)){
       c_d_img[id  + 0] = 0xFF / 2;
       c_d_img[id  + 1] /= 2;
       c_d_img[id  + 2] /= 2;
  }
}

__global__
void kernel_symmetry(unsigned int* c_d_img,unsigned int* c_d_tmp, int height, int width){
   
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;
  int id,id2;

  id  = ((  line * width  ) + col ) * 3;
  id2  = (( (height-line) * width  ) + col ) * 3;


  if ((line < height / 2 ) && (col < width )) {
       c_d_img[id  + 0] = c_d_tmp[id2  + 0];
       c_d_img[id  + 1] = c_d_tmp[id2  + 1];
       c_d_img[id  + 2] = c_d_tmp[id2  + 2];
  }
}

__global__
void kernel_grey(unsigned int* c_d_img,unsigned int* c_d_tmp, int height, int width){
   
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;
  int id,gray;

  id  = ((  line * width  ) + col ) * 3;
  gray = c_d_img[id+0]*0.299 + c_d_img[id+1]*0.587 + c_d_img[id+2]*0.114;

  if ((line < height -1 ) && (col < width -1 )) {
       c_d_img[id  + 0] = gray;
       c_d_img[id  + 1] = gray;
       c_d_img[id  + 2] = gray;
  }
}


__global__
void kernel_blur(unsigned int* c_d_img,unsigned int* c_d_tmp, int height, int width){
   
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;
  int id;

  id  = ((  ( line +1 ) * width  ) + col +1 ) * 3;
  
  
  for (int i = 0; i < 3 ; i++ ) 
  {
  int aa = c_d_img[id + ( width - 1 ) * 3 + i]; 
  int ab = c_d_img[id + ( width + 0 ) * 3 + i]; 
  int ac = c_d_img[id + ( width + 1 ) * 3 + i]; 
  int ba = c_d_img[id - 3 + i]; 
  int bc = c_d_img[id + 3 + i];
  int ca = c_d_img[id - (width - 1 ) * 3 + i]; 
  int cb = c_d_img[id - (width - 0 ) * 3 + i];
  int cc = c_d_img[id - (width + 1 ) * 3 + i];

  int moy = (aa + ab + ac + ba  + bc + ca + cb + cc)/8;
  
  if ((line < height -1 ) && (col < width -1 )) {
       c_d_img[id  + i] = moy;
  }
  }
}

__global__
void kernel_sobel(unsigned int* c_d_img,unsigned int* c_d_tmp, int height, int width){
   
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;
  int id;
  double sob;
  double sobelF[3];

  if ((col < width ) && (line < height)){

  id  = ( (line + 1 ) * width + (col +1) ) * 3;
  for (int i = 0; i < 3 ; i++ ) 
  {
  int aa = c_d_img[id + ( width - 1 ) * 3 + i];
  int ab = c_d_img[id + ( width + 0 ) * 3 + i];
  int ac = c_d_img[id + ( width + 1 ) * 3 + i];
  int ba = c_d_img[id - 3 + i];
  int bc = c_d_img[id + 3 + i];
  int ca = c_d_img[id - (width - 1 ) * 3 + i]; 
  int cb = c_d_img[id - (width - 0 ) * 3 + i];
  int cc = c_d_img[id - (width + 1 ) * 3 + i];

  int deltaX = -aa + ac   - 2*ba + 2*bc - ca  + cc;
  int deltaY = +cc + 2*cb + ca   - ac   -2*ab - aa;

  sobelF[i] = sqrt((float)(deltaX*deltaX+deltaY*deltaY));

  }

  sob = (sobelF[0] + sobelF[1] + sobelF[2])/3;

  if(sob > THRESHOLD){
    c_d_img[id + 0] = 255;
    c_d_img[id + 1] = 255;
    c_d_img[id + 2] = 255;
  }
  else
  {
    c_d_img[id + 0] = 0;
    c_d_img[id + 1] = 0;
    c_d_img[id + 2] = 0;
  }

}
}

__global__
void kernel_popArt(unsigned int* c_d_img,unsigned int* c_d_tmp, int height, int width){
   
  int col   = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;
  int id;
  id = ((line * width) + col)*3;

  if ((line < height/2 ) && (col < width/2 )){
       c_d_img[id  + 0] /= 2;
       c_d_img[id  + 1] /= 4;
       c_d_img[id  + 2] = 0xFF / 1.5;
  }

    if ((line >height/2 -1) &&(line < height ) && (col < width/2 )){
       c_d_img[id  + 0] = 0xFF - c_d_img[id + 0];
       c_d_img[id  + 1] = 0xFF / 2;
       c_d_img[id  + 2] /= 4;
  }

    if ((line > height/2 -1 ) && (line < height) && (col < width ) && (col > width/2 -1))
    {
       c_d_img[id  + 0] = 0xFF / 2;
       c_d_img[id  + 1] /= 2;
       c_d_img[id  + 2] /= 2;
  }

  int gray = c_d_img[id+0]*0.299 + c_d_img[id+1]*0.587 + c_d_img[id+2]*0.114;

  if ((line < height/2  ) && (col > width/2 ) && (col < width )) {
       c_d_img[id  + 0] = gray;
       c_d_img[id  + 1] = gray;
       c_d_img[id  + 2] = gray;
  }
}

int main (int argc , char** argv)
{
  FreeImage_Initialise();
  const char *PathName = "img.jpg";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);

  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  unsigned int *img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);
  unsigned int *d_img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);
  unsigned int *d_tmp = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);

  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  memcpy(d_img, img, 3 * width * height * sizeof(unsigned int));
  memcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int));

  unsigned int  *c_d_img, *c_d_tmp;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc((void **)&c_d_img, sizeof(unsigned int) * width * height * 3);
  cudaMalloc((void **)&c_d_tmp, sizeof(unsigned int) * width * height * 3);

  cudaMemcpy(c_d_img, img, sizeof(unsigned int) * width * height * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d_tmp, img, sizeof(unsigned int) * width * height * 3, cudaMemcpyHostToDevice);


  int nbBlocksx = width / BLOCK_WIDTH;
  if( width % BLOCK_WIDTH ) nbBlocksx++;

  int nbBlocksy = height / BLOCK_WIDTH;
  if( height % BLOCK_WIDTH ) nbBlocksy++;

  dim3 gridSize(nbBlocksx, nbBlocksy);
  dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);

  // Kernel
  cudaEventRecord(start);
  //kernel_saturation<<< gridSize , blockSize >>>(c_d_img,c_d_tmp,height,width);
  //kernel_symmetry<<< gridSize , blockSize >>>(c_d_img,c_d_tmp,height,width);
  //kernel_grey<<< gridSize , blockSize >>>(c_d_img,c_d_tmp,height,width);
  kernel_blur<<< gridSize , blockSize >>>(c_d_img,c_d_tmp,height,width);
  //kernel_sobel<<< gridSize , blockSize >>>(c_d_img,c_d_tmp,height,width);
  //kernel_popArt<<< gridSize , blockSize >>>(c_d_img,c_d_tmp,height,width);
  
  cudaEventRecord(stop);

  cudaMemcpy(d_img, c_d_img, sizeof(unsigned int) * width * height * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_tmp, c_d_tmp, sizeof(unsigned int) * width * height * 3, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);

  // Copy back
  memcpy(img, d_img, 3 * width * height * sizeof(unsigned int));

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Matrice %dx%d\n\tTemps: %f s\n", width, height, milliseconds/1000);


  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
    // next line
    bits += pitch;
  }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !

  free(img);
  free(d_img);
  free(d_tmp);
  cudaFree(c_d_img);
  cudaFree(c_d_tmp);
}

