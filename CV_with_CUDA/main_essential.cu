#include <iostream>
#include <vector>
// #include <filesystem>
#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cv_kernels.cuh"
#include "cv_methods.hpp"

#define CUDA_BLOCK_SIZE 16
#define KERNEL_SIZE 5 // must be odd

int main(void)
{
    std::string imagePath = "../Lenna.png";
    std::string outputPath = "./output.png"
    cv::Mat h_src;
    cv::Mat h_dst;
    unsigned char* d_src;
    unsigned char* d_dst;
    CVMethods cvMethod = SOBEL;

    //TODO
    // test code for batch
    // std::vector<cv::Mat> imageVec(numImages);

    h_src = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    int width = src.cols;
    int height = src.rows;
    int step = static_cast<int>(src.step);

    cudaMalloc(&d_src, height, width, CV_8UC1);
    cudaMalloc(&d_dst, height, width, CV_8UC1);
    // H2D
    cudaMemcpy(d_src, src.data, height * step, cudaMemcpyHostToDevice);

    dim3 block(CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE);
    dim3 grid((width + (CUDA_BLOCK_SIZE - 1)) / CUDA_BLOCK_SIZE, (height + (CUDA_BLOCK_SIZE - 1)) / CUDA_BLOCK_SIZE)

    // Kernel
    switch (cvMethod)
    {
    case SOBEL:
        sobelKernel <<<grid, block, 0>>> (d_src, d_dst, width, height, step);
        break;

    case MEAN:
        meanFilterKernel <<<grid, block, 0>>> (d_src, d_dst, width, height, step, KERNEL_SIZE);
        break;

    case MEDIAN:
        medianFilterKernel <<<grid, block, 0>>> (d_src, d_dst, width, height, step, KERNEL_SIZE);
        break;

    default:
        std::cout << "undefined cv method. Shut the program down." << std::endl;
        // memory free
        cudaFree(d_src);
        cudaFree(d_dst);
        retrun -1;
    }

    // D2H
    cudaMemcpy(h_dst.data, d_dst, height * step, cudaMemcpyDeviceToHost);

    // save
    cv::imwrite(outputPath, dst);

    // memory free
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
