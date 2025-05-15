#include <iostream>
#include <vector>
// #include <filesystem>
#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cv_kernels.cuh"
#include "../cv_methods.hpp"

#define CUDA_BLOCK_SIZE 16
#define KERNEL_SIZE 5 // must be odd
#define STREAM_NUM 10

int cudaEssential(void);
int streamForConcurrency(void);

int main(void)
{
    cudaEssential();
    streamForConcurrency();
}

int cudaEssential(void) 
{
    std::string imagePath = "../Lenna.png";
    std::string outputPath = "./output.png"
    cv::Mat h_src;
    cv::Mat h_dst;
    unsigned char* d_src;
    unsigned char* d_dst;
    CVMethods cvMethod = SOBEL;

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

int streamForConcurrency(void)
{
    int numImages 50;
    std::string imagePath = "../Lenna.png";
    std::string outputPath = "./output.png";
    // need to something for "*PathVec"
    std::vector<std::string> imagePathVec(numImages);
    std::vector<std::string> outputPathVec(numImages);
    std::vector<cv::Mat> h_srcVec(numImages);
    std::vector<cv::Mat> h_dstVec(numImages);
    std::vector<unsigned char*> d_srcVec(numImages);
    std::vector<unsigned char*> d_dstVec(numImages);
    // for stream management(concurrency)
    std::vector<cudaStream_t> streamVec(STREAM_NUM);
    // if you want to manage upload(H2D) or download(D2H)
    // cudaStream_t uploadStream;
    // cudaStream_t downloadStream;
    
    for (int i = 0; i < numImages; i++){
        h_srcVec[i] = cv::imread(imagePathVec[i], cv::IMREAD_GRAYSCALE);
    }
    
    int width = h_srcVec[0].cols;
    int height = h_srcVec[0].rows;
    int step = static_cast<int>(h_srcVec[0].step);
    
    for (int i = 0; i < numImages; i++) {
        cudaMalloc(&d_srcVec[i], height, width, CV_8UC1);
        cudaMalloc(&d_dstVec[i], height, width, CV_8UC1);
        // H2D
        cudaMemcpy(d_srcVec[i], h_srcVec.data, height * step, cudaMemcpyHostToDevice);
    }
    
    dim3 block(CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE);
    dim3 grid((width + (CUDA_BLOCK_SIZE - 1)) / CUDA_BLOCK_SIZE, (height + (CUDA_BLOCK_SIZE - 1)) / CUDA_BLOCK_SIZE)
    
    // for concurrency
    // Excuted the entire data(images) in units of `subIter`
    int subIterNumByStream = (numImages % STREAM_NUM == 0) ? (numImages / STREAM_NUM) : (numImages / STREAM_NUM + 1);
    int subIterInit = 0;
    int subIterEnd = STREAM_NUM;
    int subStreamIdx = 0;

    for (int i = 0; i < subIterNumByStream; i++) {
        for (int j = subInit; j < subEnd; j++) {
            subStreamIdx = j - (STREAM_NUM * i);
            if (j < numImages) {
                sobelKernel << <grid, block, 0, streamVec[subStreamIdx] >> > (d_srcVec[j], d_dstVec[j], width, height, step);
            }
        }
        subIterInit += STREAM_NUM;
        subIterEnd = (subIterEnd + STREAM_NUM <= numImages) ? (subIterEnd + STREAM_NUM) : numImages;
    }

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamDestroy(streamVec[i]);
    }

    for (int i = 0; i < numImages; i++) {
        // D2H
        cudaMemcpy(h_dstVec[i].data, d_dstVec[i], height * step, cudaMemcpyDeviceToHost);
        
        // save
        cv::imwrite(outputPathVec[i], h_dstVec[i]);
        
        // memory free
        cudaFree(d_srcVec[i]);
        cudaFree(d_dstVec[i]);
    }

    return 0;
}
