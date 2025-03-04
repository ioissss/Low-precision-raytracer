# Ray Tracer

## Introduction
基于CUDA和OpenGL的光线追踪渲染器，支持半精度浮点数和全精度浮点数运算。OpenGL用于收集GBuffer，CUDA用于执行追踪，着色，采样，降噪等所有操作。
It is a ray tracer built upon CUDA and OpenGL, working on both float16 and float32. 
OpenGL is used to generate gbuffer while CUDA is used to do anything else, including
tracing, shading, samping and denoising.

## 效果预览

![image](https://github.com/user-attachments/assets/2db5aa54-6641-43ce-99ec-9c1560c76cbe)


## Components
* glTF loader (meshes, materials, TRS animations, lights, cameras)
* scene manager
* graphic user interface
* float16 and float32 ray tracing algorithm
* shader (metallic-roughness PBR, skybox, importance sampling)
* denoiser (SVGF)

## Build (Legacy)

This software is designed to work on Windows. But it will not be so hard to port this 
software to Linux. A CUDA-capable GPU is required.

### Dependencies
* CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)
* Thrust (Shipped with CUDA Toolkit)
* GLFW3 (See instructions below)
* OpenGL (Shipped with graphics drivers usually)
* glm (See instructions below)
* half (in `third_party`)
* glad (in `third_party`)
* imgui (in `third_party`)
* tinygltf (in `third_party`)
* stb_image (in `third_party`)

Download and install the CUDA Toolkit. Refer to https://developer.nvidia.com/cuda-downloads 
for a download link and detailed instructions.

Install GLFW3 and glm manually. We recommend installing them using `vcpkg`. Refer to
https://github.com/microsoft/vcpkg for detailed instruction.

Other dependencies do not require manual installation, theoretically.

### Configuration

This project is organized by CMake. Consult Google for common knowledge about CMake
itself. The following commands introduce a possible way to configure this project.

```bash
cd ray-tracer
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=60 -G "Visual Studio 16 2019"
```
`CMAKE_CUDA_ARCHITECTURES` could be set to any value between `60` and the compute 
capability that your graphics card supports. The latter one is recommended. Please 
refer  to https://en.wikipedia.org/wiki/CUDA fora table of compute capabilities and 
corresponding graphics cards.

`-G` means generator. Invoke `cmake --help` to see a full list of supported generators. 
This software is tested on Visual Studio 16 2019, but other generators should be also
prefectly OK if you don't mind fix some potential slight compiling errors.

### Build
Double click the solution file (`.sln`) if you have adopted a Visual Studio generator.
Build this software using Visual Studio.

## Build (Visual Studio 2019/2022)
Clone the git project into a local directory. Open the directory using Visual Studio as
a CMake project. Click `Project` - `CMake Settings` and set `CMAKE_CUDA_ARCHITECTURES`.
Select `ray-tracer.exe` as the target and run it.

## Run
It is a graphic user interface application. Double click the executable or run the it
from Visual Studio directly.

Click `Free Camera` in order to move or rotate the camera. WASD,E,C is used to controll 
the position of camera. Drag while the right key of mouse is entered if you want to change 
the orientation of the camera.






