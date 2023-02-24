#include <iostream>
#include <string.h>
#include <cstdlib>
#include <cstdio>
#include "FreeImage.h"
#include <cuda_runtime.h>

// width 1920 Height 1024
#define WIDTH 3840
#define HEIGHT 2160
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

//Question 6

/* saturate_component is the GPU kernel that will run on each pixel in the image. it 
takes input a pointer to the image on the GPU, its width, height.*/
__global__ void saturate_component(unsigned int* c_d_img, int width, int height, int component) {

    /* Each thread corresponds to one pixel in the image. The calculation of the x and y coordinates 
    of the pixel is done from the thread and block identifier, using the formula blockIdx.x * blockDim.x + threadIdx.x for the x axis and 
    blockIdx.y * blockDim.y + threadIdx.y for the y axis.*/
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    /* We then check if the pixel is inside the image by comparing its x and y coordinates with the width and height of the image, if it's
    the case we saturate the pixel by putting its value to 255.*/
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

/*  In the CUDA kernel horizontal_flip, each thread retrieves the corresponding pixel values in the first and last column of the image, and then swaps
 their positions.*/
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
    /* We use the __syncthreads() function to synchronize all threads before continuing to process the image.*/
    __syncthreads();
}

//Question 8

/*The kernel blur takes as input an array c_d_img representing the original image, width width and height height of the image. 
For each pixel in the image, the kernel calculates the average value of the direct neighboring pixels taking into account the red, green 
and blue component of each pixel. The average value is then stored in the c_d_img table for each pixel.*/
__global__ void blur(unsigned int* c_d_img, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    int a1, a2, a3, b1, b2, c1, c2, c3;
    a1 = a2 = a3 = b1 = b2 = c1 = c2 = c3 = 0;

    // Calculate the values for the pixels surrounding the current pixel
    if (y > 0 && x > 0) a1 = c_d_img[idx - (width + 1) * 3];
    if (y > 0) a2 = c_d_img[idx - width * 3];
    if (y > 0 && x < width - 1) a3 = c_d_img[idx - (width - 1) * 3];
    if (x > 0) b1 = c_d_img[idx - 3];
    if (x < width - 1) b2 = c_d_img[idx + 3];
    if (y < height - 1 && x > 0) c1 = c_d_img[idx + (width - 1) * 3];
    if (y < height - 1) c2 = c_d_img[idx + width * 3];
    if (y < height - 1 && x < width - 1) c3 = c_d_img[idx + (width + 1) * 3];

    // Calculate the average color value for the surrounding pixels
    int moy_r = (a1 + a2 + a3 + b1 + b2 + c1 + c2 + c3) / 8;
    int moy_g = (a1 + a2 + a3 + b1 + b2 + c1 + c2 + c3 + 1) / 8;
    int moy_b = (a1 + a2 + a3 + b1 + b2 + c1 + c2 + c3 + 2) / 8;

    // Set the new color value for the current pixel
    c_d_img[idx] = moy_r;
    c_d_img[idx + 1] = moy_g;
    c_d_img[idx + 2] = moy_b;
    

}

//Question 9

/* The grayscale kernel allows you to gray the loaded image by applying the following formula on each pixel 
grey_value = 0.299 * red + 0.587 * green + 0.114 * blue;*/
__global__ void grayscale(unsigned int* c_d_img, int width, int height) {
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned int red = c_d_img[idx];
        unsigned int green = c_d_img[idx + 1];
        unsigned int blue = c_d_img[idx + 2];

        // Compute grayscale value
        unsigned int grey_value = 0.299 * red + 0.587 * green + 0.114 * blue;

        // Set each component of the pixel to the grayscale value
        c_d_img[idx] = grey_value;
        c_d_img[idx + 1] = grey_value;
        c_d_img[idx + 2] = grey_value;
    }
}

//Question 10

/* This kernel performs the convolution of the input image with two Sobel filters in x and y to calculate the gradient. 
The gradient is then used to calculate the magnitude of the gradient. If the magnitude is greater than a threshold (here 128), the pixel is
considered an outline and it is defined in white in the output image.*/
__global__ void sobel(unsigned int* c_d_img, int width, int height)
{
    // Sobel filter coefficients
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = (y * width + x) * 3;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {

        int gx = 0, gy = 0;

        // Compute the gradient in the x direction
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                gx += sobel_x[i+1][j+1] * c_d_img[((y + i) * width + x + j) * 3];
            }
        }

        // Compute the gradient in the y direction
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                gy += sobel_y[i+1][j+1] * c_d_img[((y + i) * width + x + j) * 3];
            }
        }

        // Compute the magnitude of the gradient
        int magnitude = abs(gx) + abs(gy);

        // Set the pixel value to white if the magnitude is greater than a threshold
        c_d_img[idx] = (magnitude > 128 ? 255 : 0);
        c_d_img[idx + 1] = (magnitude > 128 ? 255 : 0);
        c_d_img[idx + 2] = (magnitude > 128 ? 255 : 0);
    }
}

//Question 12

/* The popArt kernel takes as input an array of unsigned integers "c_d_img" that stores the loaded image, the width
and height of the image, and applies different effects to it depending on the position of the pixels in the image.*/
__global__ void popArt(unsigned int *c_d_img, int width, int height)
{
    int x  = threadIdx.x + blockDim.x * blockIdx.x;
    int y  = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = (y * width + x)*3;

    /* If the pixel is in the upper left quadrant of the image (y < height/2 and x < width/2), the function divides
    the red channel value by 2, the green channel value by 4, and sets the blue channel value to 0xFF/1.5 (about 170).*/
    if ((y < height/2 ) && (x < width/2 )){
       c_d_img[idx] /= 2;
       c_d_img[idx + 1] /= 4;
       c_d_img[idx + 2] = 0xFF / 1.5;
    }

    /* If the pixel is in the lower left quadrant of the image (y > height/2-1 and y < height and x < width/2), the
    function sets the red channel value as the maximum value (0xFF) minus the red channel value, the green channel 
    value at 0xFF/2 (about 128), and divides the blue channel value by 4.*/
    if ((y >height/2 -1) &&(y < height ) && (x < width/2 )){
       c_d_img[idx] = 0xFF - c_d_img[idx];
       c_d_img[idx + 1] = 0xFF / 2;
       c_d_img[idx + 2] /= 4;
    }

    /* If the pixel is in the lower right quadrant of the image (y > height/2-1 and y < height and x > width/2-1 and
    x < width), the function divides the values of the red, green, and blue channels by 2.*/
    if ((y > height/2 -1 ) && (y < height) && (x < width ) && (x > width/2 -1))
    {
       c_d_img[idx] = 0xFF / 2;
       c_d_img[idx + 1] /= 2;
       c_d_img[idx + 2] /= 2;
    }

    /*  If the pixel is in the upper right quadrant of the image (y < height/2 and x > width/2 and x < width), the
    function calculates the luminance value (in grayscale) of the pixel using the formula: 0.299 * red + 0.587 * green + 0.114 * blue.
    Then, it sets the value of the red, green, and blue channels to that luminance value.*/
    unsigned int grey_value = c_d_img[idx]*0.299 + c_d_img[idx+1]*0.587 + c_d_img[idx+2]*0.114;

    if ((y < height/2  ) && (x > width/2 ) && (x < width )) {
       c_d_img[idx] = grey_value;
       c_d_img[idx + 1] = grey_value;
       c_d_img[idx + 2] = grey_value;
    }
}

//Question 14

// To have the exacte same image as shown in the exemple in question 14, we run the horizontal_flip, vertical_flip and popArt kernels.
/*  The kernel vertical_flip performs a vertical inversion of the pixels of a given image. This function takes as input an array of 
pixels representing the image, as well as the dimensions of the image (width and height).*/
__global__ void vertical_flip(unsigned int* c_d_img, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    /* verifies that the indexes are valid (x must be less than the width and y must be less than half the height). If the indexes 
    are valid the code calculates the index "idx1" and "idx2" of the pixels corresponding to "y" and "height - y - 1" respectively, 
    where "height" is the height of the image. In other words, "idx1" corresponds to the current position of the pixel in the image,
    while "idx2" corresponds to the position of the pixel that is directly below it.*/
    if (x < width && y < height/2)
    {   
        int idx1 = (y * width + x) * 3;
        int idx2 = ((height - y - 1) * width + x) * 3;
        
        // Swap pixel values between idx1 and idx2
        unsigned int tmp;
        tmp = c_d_img[idx1]; c_d_img[idx1] = c_d_img[idx2]; c_d_img[idx2] = tmp;
        tmp = c_d_img[idx1+1]; c_d_img[idx1+1] = c_d_img[idx2+1]; c_d_img[idx2+1] = tmp;
        tmp = c_d_img[idx1+2]; c_d_img[idx1+2] = c_d_img[idx2+2]; c_d_img[idx2+2] = tmp;
    }
    /* We use the __syncthreads() function to synchronize all threads before continuing to process the image.*/
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
  
  unsigned int  *c_d_img, *c_d_tmp;

  cudaMalloc((void **)&c_d_img, sizeof(unsigned int) * width * height * 3);
  cudaMalloc((void **)&c_d_tmp, sizeof(unsigned int) * width * height * 3);

  cudaMemcpy(c_d_img, img, sizeof(unsigned int) * width * height * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d_tmp, img, sizeof(unsigned int) * width * height * 3, cudaMemcpyHostToDevice);


  // Kernel
  dim3 block_size(32, 32);
  dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

  //saturate_component<<<grid_size, block_size>>>(c_d_img, width, height, 0);
  //horizontal_flip<<<grid_size, block_size>>>(c_d_img, width, height);
  blur<<<grid_size, block_size>>>(c_d_img, WIDTH, HEIGHT);
  //grayscale<<<grid_size, block_size>>>(c_d_img, WIDTH, HEIGHT);
  //sobel<<<grid_size, block_size>>>(c_d_img, WIDTH, HEIGHT);
  //vertical_flip<<<grid_size, block_size>>>(c_d_img, width, height);
  //popArt<<<grid_size, block_size>>>(c_d_img, width, height);
	

  cudaMemcpy(d_img, c_d_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost);
  //cudaMemcpy(d_tmp, c_d_tmp, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost);
  
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
  cudaFree(c_d_tmp);
}
