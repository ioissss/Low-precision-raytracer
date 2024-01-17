#ifndef UTILS_GLDEBUG_HPP
#define UTILS_GLDEBUG_HPP

#include <glad/glad.h>

namespace rt {

//需要打开 DEBUG_CONTEXT, 对于 OpenGL 窗口，需要调用：
// glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
void enable_gl_debug();

} // namespace rt

#endif