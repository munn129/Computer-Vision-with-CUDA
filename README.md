# Computer Vision with CUDA

## 1. Intro
This project aims to compare CV algorithms and identify the fastest one,
and develop CUDA kernels faster than OpenCV with CUDA and TBB.

### 1.1 Project Directory
* only_cpu: using only OpenCV
* only_cuda: using only CUDA. It contains stream management code.
* opencv_with_cuda: using `cv::cuda::`

### 1.2 Progress
* **Sobel**: Issues depending on the axis direction.
* **Mean**: Implementation completed
* **Median**: CUDA kernel under development, issue depending on the sorting method.

## 2. Environments
* **OS**: Windows 11
* **CPU**: intel i9-13900K
* **GPU**: RTX A5000
* **CUDA**: 12.6
* **OpenCV**: 4.10

To run this project, you must install CUDA and build OpenCV with CUDA.
Additionally, your CPU should be an Intel processor or support OneAPI for TBB.

### 2.1 CUDA
1. Install NVIDIA graphic driver -> CUDA -> cuDNN
2. Move cuDNN files to CUDA directory  
`cudnn*.h` -> CUDA/v12.6/include  
`cudnn*.lib` -> CUDA/v12.6/lib/x64

* NVIDIA graphics driver: https://www.nvidia.com/en-us/drivers/
* CUDA: https://developer.nvidia.com/cuda-toolkit
* cuDNN: https://developer.nvidia.com/cudnn

### 2.2 OpenCV with CUDA
1. Install opencv and opencv-contrib
2. build opencv using cmake

* opencv: https://github.com/opencv/opencv
* opencv-contrib: https://github.com/opencv/opencv_contrib
* cmake: https://cmake.org/download/

### 2.2.1 CMake Flags
* `BUILD_SHARED_LIBS = ON`
* `BUILD_WITH_STATIC_CRT = OFF`
* `BUILD_opencv_world = ON`
* `OPENCV_ENABLE_NONFREE = ON`
* `OPENCV_EXTRA_MODULE_PATH = 'path-of-opencv-contrib-module'`
* `OPENCV_DNN_CUDA = ON`
* `WITH_CUDA = ON`
* `WITH_CUDNN = ON`
* `WITH_CUBLAS = ON`
* `WITH_CUFFT = ON`
* `CUDA_FAST_MATH = ON`

> **Note:**
> You may need to configure first to see all options.
> Depending on your hardware specifications (such as GPU compute capability, available memory, or installed drivers), additional configuration may be required.
> Be sure to check and adjust settings like `CUDA_ARCH_BIN` and other CUDA-related flags in CMake to match your system environment.

### 2.3 TBB
1. Install TBB and unzip it.

There's no need to build TBB.

> **For Windows users:**
> Add directory of tbb files to system environment variable.
> if you didn't add it, Visual Studio can not find dll files.

* TBB: https://github.com/uxlfoundation/oneTBB/releases

### 2.4 libtorch(not yet)
1. Install libtorch and unzip

Libtorch also doesn’t need to be built.

> **For Windows users:**
> Add the directory containing torch files to your system environment variables as well.

* libtorch: https://pytorch.org/get-started/locally/

## 3. Experiments

### 3.1 Sobel
To be updated

### 3.2 Median filter
To be updated

### 3.3 Mean filter
To be updated