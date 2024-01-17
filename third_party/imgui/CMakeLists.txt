add_library(
    ImGui 
    ./backends/imgui_impl_glfw.cpp
    ./backends/imgui_impl_opengl3.cpp
    ./backends/imgui_impl_win32.cpp
    ./imgui.cpp
    ./imgui_demo.cpp
    ./imgui_draw.cpp
    ./imgui_tables.cpp
    ./imgui_widgets.cpp
    ./misc/cpp/imgui_stdlib.cpp
    ./misc/fonts/binary_to_compressed_c.cpp
)

find_package(GLFW3 REQUIRED CONFIG)
find_package(GLFW3 REQUIRED)
target_link_libraries(ImGui glfw)

target_include_directories(ImGui PUBLIC .)