# APM
This repository contains a mini project on CUDA(Compute Unified Device Architecture). The code is a C++ program that uses CUDA to perform image processing tasks on a GPU. It includes the necessary headers and libraries for CUDA, as well as the FreeImage library for loading and saving images.
The program defines constants for the width, height, and bits per pixel of the image to be processed. It then defines six CUDA kernels: saturate_component, horizontal_flip, blur, vertical_flip, sobel and popArt which are responsible for performing different image processing tasks.
saturate_component is a kernel that takes an image on the GPU and saturates a single color component of each pixel. The kernel is launched with a grid of blocks, where each block contains a number of threads. Each thread corresponds to one pixel in the image, and calculates the x and y coordinates of the pixel using its block and thread identifier. If the pixel is inside the image, the kernel saturates the specified color component of the pixel by setting its value to 255.
horizontal_flip is a kernel that flips an image horizontally. The kernel is also launched with a grid of blocks, where each block contains a number of threads. Each thread retrieves the corresponding pixel values in the first and last column of the image, and then swaps their positions. The kernel uses __syncthreads() to synchronize all threads before continuing to process the image.
blur is a kernel that blurs an image by calculating the average value of the direct neighboring pixels for each pixel in the image. The kernel is launched with a grid of blocks, where each block contains a number of threads. For each pixel in the image, the kernel calculates the average value of the red, green, and blue components of the neighboring pixels, and stores the result in the corresponding pixel of the output image.
…
Overall, this program demonstrates how CUDA can be used to perform image processing tasks on a GPU, which can provide significant speedups compared to performing the same tasks on a CPU.
