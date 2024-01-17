#include "gl_shader.hpp"
#include "memory.hpp"
#include "utils/exception.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>

namespace rt {

// SHADER 编译
GLuint compile_shader(const std::string &source, unsigned type) {

    auto vshader_source = source.c_str();

    unsigned int shader;
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &vshader_source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        throw std::exception((std::string("failed to compile shader: %s") + infoLog).c_str());
    }

    return shader;
}

GLuint get_gbuffer_shader_program(const std::string &vshader_source, const std::string &fshader_source) {
    auto vshader = compile_shader(vshader_source, GL_VERTEX_SHADER);
    auto fshader = compile_shader(fshader_source, GL_FRAGMENT_SHADER);

    GLuint program;
    program = glCreateProgram();
    glAttachShader(program, vshader);
    glAttachShader(program, fshader);
    glLinkProgram(program);

    glDeleteShader(vshader);
    glDeleteShader(fshader);

    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        throw std::exception((std::string("failed to link shader program: ") + infoLog).c_str());
    }

    return program;
}

static std::string vertex_shader_source = R"(
    #version 460 core
    
    layout (location = 0) in vec3 aPosition;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec3 aTangent;
    layout (location = 3) in vec3 aColor;
    layout (location = 4) in vec2 aUV0;
    layout (location = 5) in vec2 aUV1;

    out vec3 world_position;
    out vec3 world_normal;
    out vec3 world_tangent;
    out vec2 uv0;
    out vec2 uv1;
    out vec3 color;
    out float clip_depth;
    
    uniform mat4 world_to_clip;
    uniform mat4 local_to_world;
    
    void main() {
        vec4 position_world4 = local_to_world * vec4(aPosition, 1.0);
        gl_Position = world_to_clip * position_world4;
        
        world_position = position_world4.xyz / position_world4.w;
        world_normal = (local_to_world * vec4(aNormal, 0.0)).xyz;
        world_tangent = (local_to_world * vec4(aTangent, 0.0)).xyz;
        uv0 = aUV0;
        uv1 = aUV1;
        color = aColor;

        clip_depth = gl_Position.z / gl_Position.w;;
    }

)";

static std::string fragment_shader_source = R"(
    #version 460 core

    in vec3 world_position;
    in vec3 world_normal;
    in vec3 world_tangent;
    in vec2 uv0;
    in vec2 uv1;
    in vec3 color;
    in float clip_depth;

    uniform uint uObjectID;

    // leave w component empty..
    layout (location = 0) out vec4 out_color;
    layout (location = 1) out uint out_objectid;
    layout (location = 2) out vec4 out_normal_depth;
    layout (location = 3) out vec4 out_position; 
    layout (location = 4) out vec4 out_tangent;
    layout (location = 5) out vec4 out_uv0uv1;
    layout (location = 6) out uint out_inobject_offset;

    void main() {
        out_color = vec4(color, 1);
        out_objectid = uObjectID;
        out_normal_depth = vec4(normalize(world_normal), clip_depth);   
        out_position = vec4(world_position, 1); 
        out_tangent = vec4(normalize(world_tangent), 1); 
        out_uv0uv1 = vec4(uv0, uv1);
        out_inobject_offset = gl_PrimitiveID * 3;
    } 
)";

GLuint get_gbuffer_shader() {
    thread_local auto shader = get_gbuffer_shader_program(vertex_shader_source, fragment_shader_source);
    return shader;
}

} // namespace rt
