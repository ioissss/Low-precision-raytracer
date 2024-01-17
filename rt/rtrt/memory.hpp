// 各种用于渲染的内存资源的管理
// 包括：GBuffer（CUDA Surface Object）、CUDA Texture Object、
// CUDA Array Resource（VBO、EBO）、CUDA 内存（thrust）、CPU 资源以及以上资源的 handle

#ifndef RTRT_GBUFFER_HPP
#define RTRT_GBUFFER_HPP

#include <glad/glad.h>

#include <array>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gl_shader.hpp"
#include "math/matrix.hpp"
#include "trace/object_bvh.hpp"
#include "utils/exception.hpp"

namespace rt {

#pragma pack(push, 1)
template <typename DataT> struct RDVertex {
    Vec3<DataT> position{0, 0, 0};
    Vec3<DataT> normal{0, 1, 0};
    Vec3<DataT> tangent{1, 0, 0};
    Vec3<DataT> color{1, 1, 1};
    Vec2<DataT> uv0{0, 0};
    Vec2<DataT> uv1{0, 0};

    RDVertex() = default;
    template <typename InT>
    RDVertex(const RDVertex<InT> &in)
        : position(in.position), normal(in.normal), tangent(in.tangent), color(in.color), uv0(in.uv0),
          uv1(in.uv1) {}
};
#pragma pack(pop)

template <typename DataT> struct RDLight {
    enum class LightType { SPOT, POINT, DIRECTIONAL } type; // all
    DataT inner_cone_angle;                                 // spot
    DataT outer_cone_angle;                                 // spot
    DataT maximum_distance = 1e5;                           // spot & point
    Vec3<DataT> direction;                                  // spot & directional
    Vec3<DataT> intensity;                                  // all
    Vec3<DataT> position;                                   // spot & point
};

struct CUDASurfaceObjectWrapper {
    CUDASurfaceObjectWrapper(cudaGraphicsResource_t *p_res, std::atomic_bool *used)
        : p_res(p_res), used(used), moved(false) {
        bool expected = false;
        bool succ = used->compare_exchange_strong(expected, true);
        if (!succ) {
            throw std::exception("allocate the same resource before release the previous one.");
        }

        RT_CHECK_CUDA(cudaGraphicsMapResources(1, p_res, 0));
        cudaResourceDesc desc;
        desc.resType = cudaResourceTypeArray;
        RT_CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&desc.res.array.array, *p_res, 0, 0));
        RT_CHECK_CUDA(cudaCreateSurfaceObject(&obj, &desc));
    }

    ~CUDASurfaceObjectWrapper() {
        if (moved)
            return;
        RT_CHECK_CUDA(cudaDestroySurfaceObject(obj));
        RT_CHECK_CUDA(cudaGraphicsUnmapResources(1, p_res, 0));
        used->store(false);
    }

    __device__ __host__ cudaSurfaceObject_t get() { return obj; }
    CUDASurfaceObjectWrapper(const CUDASurfaceObjectWrapper &) = delete;
    void operator=(const CUDASurfaceObjectWrapper &) = delete;
    CUDASurfaceObjectWrapper(CUDASurfaceObjectWrapper &&other)
        : obj(other.obj), p_res(other.p_res), used(other.used), moved(false) {
        assert(!other.moved);
        other.moved = true;
    }

  protected:
    cudaGraphicsResource_t *p_res;
    cudaSurfaceObject_t obj;
    std::atomic_bool *used;
    bool moved = false;
};

// 既可以做 RenderBuffer，也可以做 CudaSurface
class CUDASurface {
    GLuint buffer_id;
    cudaGraphicsResource_t resource_id;
    std::atomic_bool resource_used{false};
    GLenum attachment_id;
    bool use_texture;

  public:
    GLenum get_attachment_id() { return attachment_id; }
    GLuint get_buffer_id() { return buffer_id; }

    CUDASurface(int width, int height, GLuint framebuffer, GLenum format, GLenum attach_to,
                bool use_texture = false)
        : use_texture(use_texture) {
        attachment_id = attach_to;

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        if (use_texture) {
            glGenTextures(1, &buffer_id);
            glBindTexture(GL_TEXTURE_2D, buffer_id);
            glTexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
            glFramebufferTexture2D(GL_FRAMEBUFFER, attach_to, GL_TEXTURE_2D, buffer_id, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            RT_CHECK_CUDA(cudaGraphicsGLRegisterImage(&resource_id, buffer_id, GL_TEXTURE_2D,
                                                      cudaGraphicsRegisterFlagsNone));
        } else {
            glGenRenderbuffers(1, &buffer_id);
            glBindRenderbuffer(GL_RENDERBUFFER, buffer_id);
            glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, attach_to, GL_RENDERBUFFER, buffer_id);
            glBindRenderbuffer(GL_RENDERBUFFER, 0);
            RT_CHECK_CUDA(cudaGraphicsGLRegisterImage(&resource_id, buffer_id, GL_RENDERBUFFER,
                                                      cudaGraphicsRegisterFlagsNone));
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    ~CUDASurface() {
        RT_CHECK_CUDA(cudaGraphicsUnregisterResource(resource_id));

        if (use_texture) {
            glDeleteTextures(1, &buffer_id);
        } else {
            glDeleteRenderbuffers(1, &buffer_id);
        }
    }

    // Note: 持有 CUDASurfaceObjectWrapper 期间，不能用 OpenGL 访问！
    CUDASurfaceObjectWrapper get_surface_object_wrapper() {
        return CUDASurfaceObjectWrapper(&resource_id, &resource_used);
    }

    CUDASurface(const CUDASurface &) = delete;
    void operator=(const CUDASurface &) = delete;
};

/**
 * 对 OpenGL 的 frame buffer 的包装，作为 G-Buffer 使用
 */
class GBuffer {
    GLuint frame_buffer; // 绑定所有的 render buffer
    GLuint depth_buffer; // 16/32 bit 不对外暴露
    static constexpr size_t COUNT_RENDER_BUFFERS = 7;
    std::array<std::unique_ptr<CUDASurface>, COUNT_RENDER_BUFFERS> render_buffers; // 对外暴露

  public:
    int width;
    int height;
    bool half_float;

    GLuint get_framebuffer_id() { return frame_buffer; }

    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

        GLenum attachments[COUNT_RENDER_BUFFERS];
        for (int i = 0; i < render_buffers.size(); i++) {
            attachments[i] = render_buffers[i]->get_attachment_id();
        }
        glDrawBuffers(render_buffers.size(), attachments);
    }

    void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

    ~GBuffer() {
        glDeleteFramebuffers(1, &frame_buffer);
        glDeleteRenderbuffers(1, &depth_buffer);
    }

    // 修改此处代码需要修改另外两处关联：
    // 1. 修改 gbuffer.cpp 中的 shader 代码
    // 2. 在下方添加 getter
    GBuffer(int width, int height, bool half_float) : width(width), height(height), half_float(half_float) {
        glGenFramebuffers(1, &frame_buffer);

        // depth_buffer (cannot be used in cuda.. according to spec)
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
        glGenRenderbuffers(1, &depth_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // 对其到 8byte 边界
        // color
        render_buffers[0] =
            (std::make_unique<CUDASurface>(width, height, frame_buffer, GL_RGBA8, GL_COLOR_ATTACHMENT0));

        // objectid
        render_buffers[1] =
            (std::make_unique<CUDASurface>(width, height, frame_buffer, GL_R32UI, GL_COLOR_ATTACHMENT1));

        // normal_depth
        render_buffers[2] = (std::make_unique<CUDASurface>(
            width, height, frame_buffer, half_float ? GL_RGBA16F : GL_RGBA32F, GL_COLOR_ATTACHMENT2, true));

        // position_buffer
        render_buffers[3] = (std::make_unique<CUDASurface>(
            width, height, frame_buffer, half_float ? GL_RGBA16F : GL_RGBA32F, GL_COLOR_ATTACHMENT3));

        // tangent_buffer
        render_buffers[4] = (std::make_unique<CUDASurface>(
            width, height, frame_buffer, half_float ? GL_RGBA16F : GL_RGBA32F, GL_COLOR_ATTACHMENT4));

        // uv0uv1
        render_buffers[5] = (std::make_unique<CUDASurface>(
            width, height, frame_buffer, half_float ? GL_RGBA16F : GL_RGBA32F, GL_COLOR_ATTACHMENT5));

        // inobject offset
        render_buffers[6] =
            (std::make_unique<CUDASurface>(width, height, frame_buffer, GL_R32UI, GL_COLOR_ATTACHMENT6));

        glDisablei(GL_BLEND, render_buffers[1]->get_buffer_id());
        // 调用下面的语句，就会出现奇怪的报错。。
        //    glDisablei(GL_BLEND, render_buffers[6]->get_buffer_id());
    }

    GBuffer(const GBuffer &) = delete;
    void operator=(const GBuffer &) = delete;

    // todo stream i
    CUDASurfaceObjectWrapper get_color() { return render_buffers[0]->get_surface_object_wrapper(); }
    CUDASurfaceObjectWrapper get_objectid() { return render_buffers[1]->get_surface_object_wrapper(); }
    CUDASurfaceObjectWrapper get_normal_depth() { return render_buffers[2]->get_surface_object_wrapper(); }
    CUDASurfaceObjectWrapper get_position() { return render_buffers[3]->get_surface_object_wrapper(); }
    CUDASurfaceObjectWrapper get_tangent() { return render_buffers[4]->get_surface_object_wrapper(); }
    CUDASurfaceObjectWrapper get_uv0uv1() { return render_buffers[5]->get_surface_object_wrapper(); }
    CUDASurfaceObjectWrapper get_inobject_offset() { return render_buffers[6]->get_surface_object_wrapper(); }

    // GLuint get_gl_color()           { return render_buffers[0]->get_buffer_id(); }
    // GLuint get_gl_objectid()        { return render_buffers[1]->get_buffer_id(); }
    GLuint get_gl_normal_depth() { return render_buffers[2]->get_buffer_id(); }
    // GLuint get_gl_position()        { return render_buffers[3]->get_buffer_id(); }
    // GLuint get_gl_tangent()         { return render_buffers[4]->get_buffer_id(); }
    // GLuint get_gl_uv0uv1()          { return render_buffers[5]->get_buffer_id(); }
    // GLuint get_gl_inobject_offset() { return render_buffers[6]->get_buffer_id(); }
};

struct UniformLocation {
    GLint world_to_clip_loc;
    GLint local_to_world_loc;
    GLint uObjectID_loc;

    UniformLocation(GLuint shader) {
        world_to_clip_loc = glGetUniformLocation(shader, "world_to_clip");
        local_to_world_loc = glGetUniformLocation(shader, "local_to_world");
        uObjectID_loc = glGetUniformLocation(shader, "uObjectID");
    }
};

struct ObjectConstantData {
    glm::mat4 transform_L2W; // uniform 只有 float的。。
    glm::mat4 transform_W2L; // uniform 只有 float的。。
    uint32_t materialid;
    uint32_t objectid;

    void bind(const UniformLocation &location, uint32_t inscene_objid) {
        glUniformMatrix4fv(location.local_to_world_loc, 1, GL_FALSE, glm::value_ptr(transform_L2W));
        glUniform1ui(location.uObjectID_loc, inscene_objid);
    }
};

template <typename DataT> struct CUDAObjectConstantData {
    Mat4<DataT> transform_L2W; // uniform 只有 float的。。
    Mat4<DataT> transform_W2L; // uniform 只有 float的。。
    uint32_t materialid;
    uint32_t objectid;

    static Mat4<DataT> convert_matrix(glm::mat4 mat) {
        Mat4<DataT> out;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                out[{i, j}] = mat[j][i];
            }
        }
        return out;
    }

    CUDAObjectConstantData(const ObjectConstantData &data) {
        materialid = data.materialid;
        objectid = data.objectid;
        transform_W2L = convert_matrix(data.transform_W2L);
        transform_L2W = convert_matrix(data.transform_L2W);
    }
};

template <typename DataT> struct CUDAMaterial {
    // rtrt 不需要 ao

    Vec3<DataT> color{Vec3<DataT>(1, 1, 1)};
    Vec3<DataT> emission{Vec3<DataT>(0, 0, 0)};

    DataT metallic = 0;
    DataT roughness = 1;
    DataT anisotropy = 0;

    DataT normalmap_scale = 1; // TODO: 物体 scale 之后的法线，normal map scale 都暂不支持

    // 暂时无用
    bool double_sided = true;

    uint32_t uv_color = UINT32_MAX;
    uint32_t uv_emission = UINT32_MAX;
    uint32_t uv_metallic = UINT32_MAX;
    uint32_t uv_roughness = UINT32_MAX;
    uint32_t uv_normal = UINT32_MAX;

    uint32_t channel_roughness = 0;
    uint32_t channel_metallic = 0;

    cudaTextureObject_t tex_color;
    cudaTextureObject_t tex_emission;
    cudaTextureObject_t tex_metallic;
    cudaTextureObject_t tex_roughness;
    cudaTextureObject_t tex_normal;
};

struct RDTexture {
    uint32_t *array;
    cudaTextureObject_t tex;

    RDTexture(uint8_t *data, size_t width, size_t height, bool sRGB = false) {
        size_t pitch;
        RT_CHECK_CUDA(cudaMallocPitch((void **)&array, &pitch, sizeof(uint8_t) * 4 * width, height));

        RT_CHECK_CUDA(cudaMemcpy2D(array, pitch, data, 4 * width * sizeof(uint8_t),
                                   4 * width * sizeof(uint8_t), height, cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = array;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        resDesc.res.pitch2D.pitchInBytes = pitch;
        resDesc.res.pitch2D.width = width;
        resDesc.res.pitch2D.height = height;
        cudaTextureDesc texDesc = {};

        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.normalizedCoords = true;
        texDesc.sRGB = sRGB;

        RT_CHECK_CUDA(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr))
    }

    ~RDTexture() {
        RT_CHECK_CUDA(cudaDestroyTextureObject(tex));
        RT_CHECK_CUDA(cudaFree(array));
    }

    RDTexture(const RDTexture &) = delete;
    void operator=(const RDTexture &) = delete;
};

template <typename DataT> struct MaterialHolder {
    CUDAMaterial<DataT> constants;

    std::shared_ptr<RDTexture> tex_color;
    std::shared_ptr<RDTexture> tex_emission;
    std::shared_ptr<RDTexture> tex_ao;
    std::shared_ptr<RDTexture> tex_metallic;
    std::shared_ptr<RDTexture> tex_roughness;
    std::shared_ptr<RDTexture> tex_normal;
};

template <typename DataT> struct RDCamera {
    // 暂不支持正交投影...
    Mat4<DataT> transform;
    Mat4<DataT> transform_L2W;
    DataT field_of_view_y = M_PI / 2;
    DataT aspect_ratio = 1;
    DataT z_near = 0.1;
    DataT z_far = 100;

    RDCamera() = default;

    bool operator==(const RDCamera<DataT> &rhs) const {
        auto b1 = (transform - rhs.transform).norm2_squared() <= DataT(1e-8);
        auto b2 = field_of_view_y == rhs.field_of_view_y;
        auto b3 = aspect_ratio == rhs.aspect_ratio;
        return b1 && b2 && b3;
    }

    bool operator!=(const RDCamera<DataT> &rhs) const { return !(*this == rhs); }

    template <typename O>
    explicit RDCamera(const RDCamera<O> &o)
        : transform(o.transform), field_of_view_y(o.field_of_view_y), aspect_ratio(o.aspect_ratio) {}
};

struct BufferObject {
    GLuint index;
    cudaGraphicsResource_t cuda_resource;

    void register_cuda_resource() {
        RT_CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&cuda_resource, index, cudaGraphicsRegisterFlagsReadOnly))
    }

    void unregister_cuda_resource() { cudaGraphicsUnregisterResource(cuda_resource); }

    template <typename DataT> const DataT *get_mapped_array() {
        DataT *ptr;
        RT_CHECK_CUDA(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&ptr), nullptr, cuda_resource))
        return ptr;
    }

    const void *get_mapped_raw_array() {
        void *ptr;
        RT_CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((&ptr), nullptr, cuda_resource))
        return ptr;
    }

    void map() { RT_CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_resource)) }

    void unmap() { RT_CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_resource)) }
};

template <typename DataT> struct MeshVBO : public BufferObject {

    explicit MeshVBO(const std::vector<RDVertex<DataT>> &raw_data) {
        glGenBuffers(1, &index);
        glBindBuffer(GL_ARRAY_BUFFER, index);
        glBufferData(GL_ARRAY_BUFFER, sizeof(raw_data[0]) * raw_data.size(), raw_data.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        this->register_cuda_resource();
    }

    void bind() {
        constexpr bool half_float = sizeof(DataT) == 2;
        constexpr uint32_t FLOAT_TYPE = half_float ? GL_HALF_FLOAT : GL_FLOAT;
        constexpr uint32_t ELEMENT_SIZE = sizeof(RDVertex<DataT>);

        glBindBuffer(GL_ARRAY_BUFFER, index);

        size_t offset = 0;

        glVertexAttribPointer(0, 3, FLOAT_TYPE, false, ELEMENT_SIZE, (void *)offset);
        glEnableVertexAttribArray(0);
        offset += sizeof(RDVertex<DataT>::position);

        glVertexAttribPointer(1, 3, FLOAT_TYPE, false, ELEMENT_SIZE, (void *)offset);
        glEnableVertexAttribArray(1);
        offset += sizeof(RDVertex<DataT>::normal);

        glVertexAttribPointer(2, 3, FLOAT_TYPE, false, ELEMENT_SIZE, (void *)offset);
        glEnableVertexAttribArray(2);
        offset += sizeof(RDVertex<DataT>::tangent);

        glVertexAttribPointer(3, 3, FLOAT_TYPE, false, ELEMENT_SIZE, (void *)offset);
        glEnableVertexAttribArray(3);
        offset += sizeof(RDVertex<DataT>::color);

        glVertexAttribPointer(4, 2, FLOAT_TYPE, false, ELEMENT_SIZE, (void *)offset);
        glEnableVertexAttribArray(4);
        offset += sizeof(RDVertex<DataT>::uv0);

        glVertexAttribPointer(5, 2, FLOAT_TYPE, false, ELEMENT_SIZE, (void *)offset);
        glEnableVertexAttribArray(5);
        // offset += sizeof(VertexData<DataT>::uv1);
    }

    ~MeshVBO() {
        this->unregister_cuda_resource();
        glDeleteBuffers(1, &index);
    }
    MeshVBO(const MeshVBO &) = delete;
    void operator=(const MeshVBO &) = delete;
};

struct MeshEBO : public BufferObject {
    explicit MeshEBO(const std::vector<uint32_t> &indices) {
        glGenBuffers(1, &index);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(), indices.data(),
                     GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        this->register_cuda_resource();
    }

    void bind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index); }

    ~MeshEBO() {
        this->unregister_cuda_resource();
        glDeleteBuffers(1, &index);
    }
    MeshEBO(const MeshEBO &) = delete;
    void operator=(const MeshEBO &) = delete;
};

struct FrameBuffer {
    GLuint index;
    explicit FrameBuffer() { glGenFramebuffers(1, &index); }

    void bind() { glBindFramebuffer(GL_FRAMEBUFFER, index); }
    void bind_read() { glBindFramebuffer(GL_READ_FRAMEBUFFER, index); }
    void bind_draw() { glBindFramebuffer(GL_DRAW_FRAMEBUFFER, index); }

    ~FrameBuffer() {
        glDeleteFramebuffers(1, &index);
    }
    FrameBuffer(const FrameBuffer &) = delete;
    void operator=(const FrameBuffer &) = delete;
};

template <typename DataT> class RDResource;

struct GLMeshVAO {
    GLuint VAO = 0;

    uint32_t start_index;
    uint32_t n_indices;
    uint32_t vbo_index;
    uint32_t ebo_index;

    bool double_sided = true;

    // 绘制 Mesh
    void draw() const {
        if (!double_sided) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
            glFrontFace(GL_CCW);
        } else {
            glDisable(GL_CULL_FACE);
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, n_indices, GL_UNSIGNED_INT,
                       (void *)static_cast<size_t>(start_index * sizeof(uint32_t)));
        glBindVertexArray(0);
    }

    template <typename DataT>
    GLMeshVAO(uint32_t vbo_index, uint32_t ebo_index, uint32_t start_index, uint32_t n_indices,
              bool double_sided, RDResource<DataT> &resource);

    ~GLMeshVAO() { glDeleteVertexArrays(1, &VAO); }

    GLMeshVAO(const GLMeshVAO &) = delete;
    void operator=(const GLMeshVAO &) = delete;
};

struct CUDAMeshVAO {
    uint32_t ebo_index; // 哪个 EBO
    uint32_t vbo_index; // 哪个 VBO

    uint32_t ebo_offset; // EBO 的哪个位置（uint32_t 计）
    uint32_t ebo_length; // EBO 的多大长度
};

template <typename DataT> struct RDResource {

    typedef MeshVBO<DataT> MeshVBOType;

    std::vector<MaterialHolder<DataT>> materials;
    std::vector<std::shared_ptr<MeshEBO>> objects_ebo;
    std::vector<std::shared_ptr<MeshVBOType>> objects_vbo;

    std::vector<std::shared_ptr<GLMeshVAO>> objects_vao;
    // 场景 AABB 存储为 float32，构建 BVH 后存储为 float16
    std::vector<std::shared_ptr<ObjectBVH<DataT>>> objects_bvh;

    // hz: 坐标转换矩阵
    std::vector<Matrix<3, 3, DataT>> M_shift;
    // hz: float32
    std::vector<Vec3<float>> v_positions_f32;
    std::vector<Matrix<3, 3, float>> M_shift_f32;

    RDVertex<DataT> const *const *generate_vbos() {
        thrust::host_vector<const RDVertex<DataT> *> host_vec(objects_vbo.size());
        for (size_t i = 0; i < objects_vbo.size(); i++) {
            std::shared_ptr<MeshVBOType> vbo = objects_vbo[i];
            auto ptr = vbo->get_mapped_raw_array();
            host_vec[i] = reinterpret_cast<const RDVertex<DataT> *>(ptr);
        }
        vbos_buffer = host_vec;
        return thrust::raw_pointer_cast(vbos_buffer.data());
    }

    uint32_t const *const *generate_ebos() {
        thrust::host_vector<const uint32_t *> host_vec(objects_ebo.size());
        for (size_t i = 0; i < objects_ebo.size(); i++) {
            host_vec[i] = objects_ebo[i]->get_mapped_array<uint32_t>();
        }
        ebos_buffer = host_vec;
        return thrust::raw_pointer_cast(ebos_buffer.data());
    }

    Matrix<3, 3, DataT> *generate_m() {
        m_buffer = M_shift;
        return thrust::raw_pointer_cast(m_buffer.data());
    }

    Matrix<3, 3, float> *generate_mf32() {
        mf32_buffer = M_shift_f32;
        return thrust::raw_pointer_cast(mf32_buffer.data());
    }

    Vec3<float> *generate_positionf32() {
        position_buffer = v_positions_f32;
        return thrust::raw_pointer_cast(position_buffer.data());
    }

    void compute_m(std::vector<RDVertex<float>> vbo_data, std::vector<uint32_t> indices) {
        M_shift.resize(indices.size() / 3);
        M_shift_f32.resize(indices.size() / 3);
        v_positions_f32.resize(indices.size());

        for (int i = 0; i < indices.size();) {
            Vec3<float> v0 = vbo_data[indices[i]].position;
            Vec3<float> v1 = vbo_data[indices[i + 1]].position;
            Vec3<float> v2 = vbo_data[indices[i + 2]].position;

            v_positions_f32[indices[i]] = vbo_data[indices[i]].position;
            v_positions_f32[indices[i + 1]] = vbo_data[indices[i + 1]].position;
            v_positions_f32[indices[i + 2]] = vbo_data[indices[i + 2]].position;

            Matrix<3, 1, float> col0 = v0 - v2;
            Matrix<3, 1, float> col1 = v1 - v2;
            Matrix<3, 1, float> col2 = cross_product_difference(v0 - v2, v1 - v2) - v2;
            Matrix<3, 3, float> M1;
            M1.set_col(0, col0);
            M1.set_col(1, col1);
            M1.set_col(2, col2);

            Matrix<3, 3, float> M2 = M1.inversed_3_3();
            M_shift_f32[i / 3] = M2;
            M_shift[i / 3] = static_cast<Matrix<3, 3, DataT>>(M2);
            i = i + 3;
        }
    }

  private:
    thrust::device_vector<const RDVertex<DataT> *> vbos_buffer;
    thrust::device_vector<const uint32_t *> ebos_buffer;
    // hz
    thrust::device_vector<Matrix<3, 3, DataT>> m_buffer;
    thrust::device_vector<Matrix<3, 3, float>> mf32_buffer;
    thrust::device_vector<Vec3<float>> position_buffer;
};

template <typename DataT>
GLMeshVAO::GLMeshVAO(uint32_t vbo_index, uint32_t ebo_index, uint32_t start_index, uint32_t n_indices,
                     bool double_sided, RDResource<DataT> &resource)
    : start_index(start_index), n_indices(n_indices), vbo_index(vbo_index), ebo_index(ebo_index),
      double_sided(double_sided) {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    resource.objects_vbo[vbo_index]->bind();
    resource.objects_ebo[ebo_index]->bind();
    glBindVertexArray(0);
}

template <typename DataT> struct RDSkybox {
    std::shared_ptr<RDTexture> texture;

    // 图像空间中加减
    DataT delta_x = 0;
    DataT delta_y = 0;

    DataT exposure = 1;

    struct Ref {
        cudaTextureObject_t texture;
        
        DataT delta_x = 0;
        DataT delta_y = 0;

        DataT exposure = 1;

        bool valid = false;
    };

    Ref get_ref() { return {texture ? texture->tex: 0, delta_x, delta_y, exposure, texture != nullptr}; }
};

template <typename DataT> class RDScene {
  public:
    RDScene() = default;

    RDScene(const RDScene &) = delete;
    void operator=(const RDScene &) = delete;

    std::vector<ObjectConstantData> objects_constants;
    std::vector<std::pair<glm::vec3, glm::vec3>> objects_aabb; // in world coordinate
    std::vector<RDLight<DataT>> lights;

    RDCamera<DataT> camera;

    RDSkybox<DataT> skybox;
};

} // namespace rt

#endif