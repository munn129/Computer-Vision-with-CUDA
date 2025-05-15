#include <iostream>
#include <vector>
#include <filesystem>  // need upper version than C++ 17
#include <opencv2/opencv.hpp>
// #include <tbb/tbb.h>

#include "../cv_methods.hpp"

#define KERNEL_SIZE 5

int essential(void);

int main(void)
{
    essential(void);
}

int essential(void)
{
    std::string imagePath = "../Lenna.png";
    std::string outputPath = "./output.png"
    cv::Mat src;
    // cv::Mat dst_x; // for SOBEL
    // cv::Mat dst_y; // for SOBEL
    cv::Mat dst;
    CVMethods cvMethod = SOBEL;

    src = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    switch (cvMethod)
    {
    case SOBEL:
        cv::Sobel(src, dst, CV_8UC1, 1, 1, 3);

        /*
        // cuda kernel works like this
        cv::Sobel(src, dst_x, CV_8UC1, 1, 0, 3);
        cv::Sobel(src, dst_y, CV_8UC1, 0, 1, 3);

        cv::Mat abs_x, abs_y;
        cv::convertScaleAbs(dst_x, abs_x);
        cv::convertScaleAbs(dst_y, abs_y);
        cv::add(abs_x, abs_y, dst);
        */
        break;

    case MEAN:
        cv::blur(src, dst, cv::Size(KERNEL_SIZE, KERNEL_SIZE));
        break;

    case MEDIAN:
        cv::medianBlur(src, dst, KERNEL_SIZE);
        break;
    
    default:
        break;
    }

    // save
    cv::imwrite(outputPath, dst);
}
