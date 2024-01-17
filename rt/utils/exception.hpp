#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

namespace rt {

void check_cuda_error(cudaError_t code);

}

#define RT_CHECK_CUDA(code) rt::check_cuda_error(code);

#endif
