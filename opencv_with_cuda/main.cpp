#include <iostream>
#include <vector>
#include <filesystem>  // need upper version than C++ 17
#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <tbb/tbb.h>

#include "../cv_methods.hpp"

#define DO_TBB
#define KERNEL_SIZE 5

int essential(void);
int streamForConcurrency(void);

int main(void)
{

}

int essential(void)
{
    std::string imagePath = "../Lenna.png";
    std::string outputPath = "./output.png";
    cv::Mat h_src;
    cv::Mat h_dst;
    cv::cuda::GpuMat d_src;
    cv::cuda::GpuMat d_dst;
    cv::Ptr<cv::cuda::Filter> filter;
    CVMethods cvMethod = SOBEL;

    h_src = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // H2D
    d_src.upload(h_src);

    switch (cvMethod)
    {
    case SOBEL:
        // x and y
        filter = cv::cuda::createSobelFilter(d_src.type(), d_src.type(), 1, 1, 3);
        break;

    case MEAN:
        filter = cv::cuda::createBoxFilter(d_src.type(), d_src.type(), cv::Size(KERNEL_SIZE, KERNEL_SIZE));
        break;

    case MEDIAN:
        filter = cv::cuda::createMedianFilter(d_src.type(), KERNEL_SIZE);
        break;

    default:
        std::cout << "undefined cv method. Shut the program down." << std::endl;
        return -1;
    }

    // Kernel
    filter->apply(d_src, d_dst);

    // D2H
    d_dst.download(h_dst);

    // save
    cv::imwrite(outputPath, dst);

    return 0;
}

int streamForConcurrency(void)
{

}
