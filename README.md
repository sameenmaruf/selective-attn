This is an implementation of our paper:

Sameen Maruf, Andre F. T. Martins and Gholamreza Haffari: Selective Attention for Context-aware Neural Machine Translation. Accepted at NAACL-HLT 2019.

Please cite our paper if you use our code. 

# Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), using the development version (not release), e.g. 3.3.beta2 (http://bitbucket.org/eigen/eigen/get/3.3-beta2.tar.bz2)

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [boost](http://www.boost.org/), e.g., 1.58 using *libboost-all-dev* ubuntu package

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

# Building

First, clone the repository

git clone https://github.com/sameenmaruf/selective-attn.git

As mentioned above, you'll need the latest [development] version of eigen

hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb

A version of latest DyNet (v2.0.3) is already included (e.g., dynet folder). 

# CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH [-DBoost_NO_BOOST_CMAKE=ON]
    make -j 2

Boost note. The "-DBoost_NO_BOOST_CMAKE=ON" can be optional but if you have a trouble of boost-related build error(s), adding it will help to overcome. 

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DMKL=TRUE -DMKL_ROOT=MKL_PATH -DBoost_NO_BOOST_CMAKE=TRUE -DENABLE_BOOST=TRUE

substituting in different paths to EIGEN_PATH and MKL_PATH if you have placed them in different directories. 

This will build the following binaries
    
    build_cpu/transformer-train
    build_cpu/transformer-decode
    build_cpu/transformer-context
    build_cpu/transformer-context-decode
    build_cpu/transformer-computerep

# GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 8.0.61.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=CUDA_PATH -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DMKL=TRUE -DMKL_ROOT=MKL_PATH -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DENABLE_BOOST=TRUE
    make -j 2

substituting in your EIGEN_PATH and CUDA_PATH folders, as appropriate. Make sure to include MKL if using sparsemax as it is implemented for CPU.

This will result in the binaries

    build_gpu/transformer-train
    build_gpu/transformer-decode
    build_gpu/transformer-context
    build_gpu/transformer-context-decode
    build_gpu/transformer-computerep

# Using the model

See readme_commands.txt

# References

The original Transformer implementation used in our code is available at https://github.com/duyvuleo/Transformer-DyNet

## Contacts

Sameen Maruf (sameen.maruf@monash.edu; sameen.maruf@gmail.com)

---
Updated June 2019
