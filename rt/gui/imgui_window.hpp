#ifndef RAYTRACER_RT_GUI_IMGUI_WINDOW_HPP
#define RAYTRACER_RT_GUI_IMGUI_WINDOW_HPP

#include <cuda.h>
#include <cuda_runtime.h>

namespace rt {

struct DebugInfo {
    int window_x;
    int window_y;
    int window_width;
};


int run_imgui();


extern rt::DebugInfo debug_info;

}


#endif