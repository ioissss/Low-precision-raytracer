#if !defined(_WIN32)
#error "尚不支持非 Windows 平台，但应当容易迁移"
#endif

#include <glad/glad.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdlib.h>

#include "gui/imgui_window.hpp"
#include "utils/exception.hpp"

#include <cstdio>

using namespace std;

int main(int argc, char **argv) {

    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    //if(!GLAD_GL_ARB_half_float_pixel) {
    //    printf("half float pixel is not supported");
    //}

    //if (!GLAD_GL_ARB_half_float_vertex) {
    //    printf("half float vertex is not supported");
    //}
    srand(time(nullptr));
    srand(time(nullptr));
    rt::run_imgui();
}
