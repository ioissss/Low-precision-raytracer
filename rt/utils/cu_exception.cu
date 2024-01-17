#include "utils/exception.hpp"

#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

namespace rt {
void check_cuda_error(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::exception(cudaGetErrorString(code));
    }
}
} // namespace rt