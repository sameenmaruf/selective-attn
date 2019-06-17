#ifndef DYNET_CUDA_H
#define DYNET_CUDA_H
#if HAVE_CUDA

#include <vector>
#include <utility>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#if HAVE_CUDNN
#include <cudnn.h>
#endif
#include "dynet/except.h"
#include "dynet/devices.h"

#define MAX_GPUS 256

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      if (err == cudaErrorMemoryAllocation) {              \
        show_pool_mem_info();                              \
        size_t free_bytes=0, total_bytes=0;                \
        cudaMemGetInfo(&free_bytes, &total_bytes);         \
        int devid=-1;                                      \
        cudaGetDevice(&devid);                             \
        std::cerr << "CUDA is unable to allocate enough "  \
                  << "GPU memory on GPU:" << devid         \
                  << ", at current stage only "            \
                  << free_bytes/1024/1024 << " MB out of " \
                  << total_bytes/1024/1024 << " MB is free"\
                  << ". Note due to hardware limitations " \
                  << "not all free memories can be "       \
                  << "allocated."                          \
                  << std::endl;                            \
      }                                                    \
      throw dynet::cuda_exception(#stmt);                  \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = stmt;                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "CUBLAS failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw dynet::cuda_exception(#stmt);                  \
    }                                                      \
  } while(0)

#define CURAND_CHECK(stmt) do {                            \
    curandStatus_t stat = stmt;                            \
    if (stat != CURAND_STATUS_SUCCESS) {                   \
      std::cerr << "CURAND failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw dynet::cuda_exception(#stmt);                  \
    }                                                      \
  } while(0)

#if HAVE_CUDNN
#define CUDNN_CHECK(stmt) do {                             \
    cudnnStatus_t stat = (stmt);                           \
    if (stat != CUDNN_STATUS_SUCCESS){                     \
      std::cerr << "CUDNN failure in " << #stmt            \
                << std::endl << cudnnGetErrorString(stat)  \
                << std::endl;                              \
      throw dynet::cuda_exception(#stmt);                  \
    }                                                      \
  } while(0)
#endif

namespace dynet {

struct DynetParams;


class Device;

inline std::pair<int, int> SizeToBlockThreadPair(int n) {
  DYNET_ASSERT(n > 0, "Bad thread size in GPU code " << n);
  int logn;
#if defined(_MSC_VER)
  logn = 0;
  if (n > 2) {
    int localN = n - 1;
    while (localN >>= 1)
      logn++;
  }
#else
  asm("\tbsr %1, %0\n"
      : "=r"(logn)
      : "r" (n-1));
#endif
  logn = logn > 9 ? 9 : (logn < 4 ? 4 : logn);
  ++logn;
  int threads = 1 << logn;
  int blocks = (n + threads - 1) >> logn;
  blocks = blocks > 65535 ? 65535 : blocks;
  return std::make_pair(blocks, threads);
}

std::vector<Device*> initialize_gpu(dynet::DynetParams& params);
std::vector<Device*> initialize_gpu(int& argc, char**& argv);

} // namespace dynet

#endif
#endif
