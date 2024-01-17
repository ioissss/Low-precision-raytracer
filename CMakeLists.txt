cmake_minimum_required (VERSION 3.20)

project (ray_tracer LANGUAGES CUDA C CXX)

if (MSVC)
    add_definitions(-DSTBI_MSC_SECURE_CRT=1)
    # add_definitions(-DWXUSINGDLL=1)
endif()

# 都不好用。暂时改成 UTF-8 BOM 了。。
# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>")
# add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
# add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_subdirectory(third_party)
add_subdirectory(rt)


