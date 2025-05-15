# computer vision with CUDA

## 1. Intro
This project aims to compare CV algorithms and identify the fastest one,
and develop CUDA kernels faster than OpenCV with CUDA and TBB.

## 2. Environments
- OS: Windows 11
- CPU: intel i9-13900K
- GPU: RTX A5000
- CUDA: 12.6
- openCV: 4.10

To run this project, you must install CUDA and build OpenCV with CUDA.
Additionally, yout CPU should be an Intel processor or support OneAPI for TBB.

### 2.1 CUDA
1. Install NVIDIA graphic driver -> CUDA -> cuDNN
2. Move cuDNN files to CUDA directory

* NVIDIA graphic driver: https://www.nvidia.com/en-us/drivers/
* CUDA: https://developer.nvidia.com/cuda-toolkit
* cuDNN: https://developer.nvidia.com/cudnn

### 2.2 OpenCV with CUDA
1. Install opencv and opencv-contrib
2. build opencv using cmake

* opencv: https://github.com/opencv/opencv
* opencv-contrib: https://github.com/opencv/opencv_contrib
* cmake: https://cmake.org/download/

### 2.3 TBB
1. Install TBB and unzip

TBB don't need to build

(for Windows user) add directory of tbb.dll to system environment variable. if you didn't add it, Visual Studio can not find dll files.

* TBB: https://github.com/uxlfoundation/oneTBB/releases

### 2.4 libtorch
1. Install libtorch and unzip

Libtorch don't need to build too.

(for Windows user) add directory of torch.lib to system environment variable too.

* libtorch: https://pytorch.org/get-started/locally/

## 3. Experiments

### 3.1 Sobel
To be updated

### 3.2 Median filter
To be updated

### 3.3 Mean filter
To be updated