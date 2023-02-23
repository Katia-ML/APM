#include <iostream>
#include <string.h>
#include <cstdlib>
#include <cstdio>
#include "FreeImage.h"
#include <cuda_runtime.h>

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

//Question 6
__global__ void saturate_component(unsigned int* c_d_img, int width, int height, int component) {


    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        if (component == 0) {  // saturate red component
            c_d_img[idx] = 255;
        } else if (component == 1) {  // saturate green component
            c_d_img[idx + 1] = 255;
        } else {  // saturate blue component
            c_d_img[idx + 2] = 255;
        }
    }
}

//Question 7
__global__ void horizontal_flip(unsigned int* c_d_img, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width / 2 && y < height)
    {
        int idx1 = (y * width + x) * 3;
        int idx2 = (y * width + (width - x - 1)) * 3;
        
        // Swap pixel values between idx1 and idx2
        unsigned int tmp;
        tmp = c_d_img[idx1]; c_d_img[idx1] = c_d_img[idx2]; c_d_img[idx2] = tmp;
        tmp = c_d_img[idx1+1]; c_d_img[idx1+1] = c_d_img[idx2+1]; c_d_img[idx2+1] = tmp;
        tmp = c_d_img[idx1+2]; c_d_img[idx1+2] = c_d_img[idx2+2]; c_d_img[idx2+2] = tmp;
    }

    __syncthreads();
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

  unsigned int  *c_d_img;

  cudaMalloc((void **)&c_d_img, sizeof(unsigned int) * width * height * 3);
  cudaMemcpy(c_d_img, img, sizeof(unsigned int) * width * height * 3, cudaMemcpyHostToDevice);


  // Kernel
  dim3 block_size(32, 32);
  dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

  saturate_component<<<grid_size, block_size>>>(c_d_img, width, height, 0);
  //horizontal_flip<<<grid_size, block_size>>>(c_d_img, width, height);


  cudaMemcpy(d_img, c_d_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost);
  
  // Copy back
  memcpy(img, d_img, 3 * width * height * sizeof(unsigned int));

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
}
