set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING 
	"CUDA Compute Capability。前往 https://en.wikipedia.org/wiki/CUDA 查询。如，对于8.6，输入 86" )

if(${CMAKE_CUDA_ARCHITECTURES} LESS 60) 
	message(FATAL_ERROR "CUDA Compute Capability 设置为" ${CMAKE_CUDA_ARCHITECTURES} "。 "
		"但程序支持的最低版本是 60。 "
		"前往 https://en.wikipedia.org/wiki/CUDA 查询 GPU 支持的最高版本，通过 -DCMAKE_CUDA_ARCHITECTURES=<value> 设置")
endif()

message(NOTICE "采用的 CUDA Compute Capability：" ${CMAKE_CUDA_ARCHITECTURES})
message(NOTICE "前往 https://en.wikipedia.org/wiki/CUDA 查询 GPU 支持的最高版本，通过 -DCMAKE_CUDA_ARCHITECTURES=<value> 设置")


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DRT_DEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DRT_DEBUG")

add_definitions(-DUSE_MATH_DEFINES)
add_definitions(-DGLM_FORCE_RADIANS)

find_package(Thrust REQUIRED CONFIG)
find_package(GLFW3 REQUIRED CONFIG)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)

thrust_create_target(Thrust)

add_executable(
	ray-tracer
	main.cu
	utils/gldebug.cpp
	utils/cu_exception.cu
	gui/imgui_window.cu
	"rtrt/memory.hpp"
	rtrt/loader.cu
	rtrt/loader.hpp
	rtrt/cuda.hpp
	rtrt/gl_shader.hpp
	rtrt/gl_shader.cpp
	math/number.hpp
 "rtrt/hierarchy.hpp" "trace/object_bvh.hpp" "trace/scene_bvh.hpp" "gui/controller.hpp" "rtrt/bsdf.hpp" "rtrt/svgf.hpp")

target_include_directories(ray-tracer PRIVATE .)
target_link_libraries(ray-tracer PRIVATE tinygltf half Thrust glfw Glad ImGui OpenGL::GL glm::glm)

target_compile_options(ray-tracer PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(ray-tracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--diag-suppress;20012;--linker-options;/NODEFAULTLIB:MSVCRT>")
