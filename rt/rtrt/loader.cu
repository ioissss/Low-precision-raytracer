/// 这个 loader 更多考虑兼容性和实现的简洁程度，效率比较低

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "tiny_gltf.h"

#include <glad/glad.h>

#include <cassert>
#include <filesystem>
#include <map>

#include "hierarchy.hpp"
#include "loader.hpp"
#include "math/matrix.hpp"
#include "math/number.hpp"
#include "memory.hpp"
#include "trace/object_bvh.hpp"
#include "trace/scene_bvh.hpp"
#include "utils/exception.hpp"

#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace tinygltf;

namespace rt {

template <typename Out, typename In> std::vector<Out> convert_container(const std::vector<In> &in) {
    std::vector<Out> out;
    out.reserve(in.size());
    for (auto val : in) {
        out.push_back(Out(val));
    }
    return out;
}

template <typename DataT> struct MeshDesc {
    uint32_t material_id = 0;
    uint32_t start_index = 0;
    uint32_t n_indices = 0;

    // 构建每个 Object 的 AABB 是个离线处理的过程，所以使用 float 类型
    glm::vec3 aabb_lower;
    glm::vec3 aabb_upper;

    std::shared_ptr<ObjectBVH<DataT>> object_bvh;
};

template <typename TARGET,
          typename = std::enable_if_t<std::is_trivial_v<TARGET> && std::is_standard_layout_v<TARGET>>>
TARGET load_little_endian(const void *raw_data) {
    // todo: test it on a big-endian machine..

    union {
        uint32_t u32;
        uint8_t u4[4];
    } endian_tester;
    endian_tester.u32 = 0x01000000;
    bool is_big_endian = endian_tester.u4[0] == 1;

    if (is_big_endian && sizeof(TARGET) > 1) {
        // std::byteswap is not currently supported by gcc...
        if constexpr (sizeof(TARGET) == 4) {
            uint32_t data = *reinterpret_cast<const uint32_t *>(raw_data);
            uint32_t new_data = ((data & 0x000000FF) << 24) | ((data & 0x0000FF00) << 8) |
                                ((data & 0x00FF0000) >> 8) | ((data & 0xFF000000) >> 24);
            return *reinterpret_cast<TARGET *>(&new_data);
        } else if constexpr (sizeof(TARGET) == 2) {
            uint16_t data = *reinterpret_cast<const uint16_t *>(raw_data);
            uint16_t new_data = ((data & 0x00FF) << 8) | ((data & 0xFF00) >> 8);
            return *reinterpret_cast<TARGET *>(&new_data);
        } else {
            auto data = *reinterpret_cast<const TARGET *>(raw_data);
            auto *p_data = reinterpret_cast<uint8_t *>(&data);
            size_t size = sizeof(TARGET);
            for (int i = 0; i < size / 2; i++) {
                std::swap(p_data[i], p_data[size - 1 - i]);
            }
            return data;
        }
    } else {
        // x86
        return *reinterpret_cast<const TARGET *>(raw_data);
    }
}

template <typename TargetMatrix>
std::vector<TargetMatrix> load_data(tinygltf::Model model, int accessor_index) {

    if (accessor_index < 0) {
        throw std::exception("gltf/glb file is broken: accessor index < 0");
    }

    constexpr int M = TargetMatrix::ShapeM;
    constexpr int N = TargetMatrix::ShapeN;

    auto accessor = model.accessors[accessor_index];
    if (accessor.sparse.isSparse) {
        throw std::exception("mesh uses sparse accessor, which involves an "
                             "unimplemented feature");
    }

    auto view = model.bufferViews[accessor.bufferView];
    auto buffer = model.buffers[view.buffer];
    auto stride = accessor.ByteStride(view);
    auto count = view.byteLength / stride;

    auto component_length = 0;

    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        component_length = 1;
        break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        component_length = 2;
        break;
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        component_length = 4;
        break;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        component_length = 8;
        break;
    default:
        throw std::exception("undefined component type. corrupted glTF file.");
    }

    using TargetDataType = typename TargetMatrix::InnerDataType;

    auto loader_wrapper = [&accessor](TargetDataType *dest, const void *raw_data) {
        switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            *dest = TargetDataType(load_little_endian<int8_t>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            *dest = TargetDataType(load_little_endian<double>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            *dest = TargetDataType(load_little_endian<float>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_INT:
            *dest = TargetDataType(load_little_endian<int32_t>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_SHORT:
            *dest = TargetDataType(load_little_endian<int16_t>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            *dest = TargetDataType(load_little_endian<uint8_t>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            *dest = TargetDataType(load_little_endian<uint32_t>(raw_data));
            break;

        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            *dest = TargetDataType(load_little_endian<uint16_t>(raw_data));
            break;

        default:
            throw std::exception("undefined component type. corrupted glTF file.");
        }
    };

    std::vector<TargetMatrix> out;

    for (int index = 0; index < count; index++) {
        size_t offset = view.byteOffset + accessor.byteOffset + stride * index;
        TargetMatrix mat;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                loader_wrapper(&mat.data[i][j], &buffer.data[offset]);
                offset += component_length;
            }
        }
        out.push_back(mat);
    }

    return out;
}

std::vector<Vec3f> load_mesh_position(const tinygltf::Model &model, const tinygltf::Primitive &primitive) {

    if (!primitive.attributes.count("POSITION")) {
        throw std::exception("mesh does not have attribute POSITION");
    }

    auto attr = primitive.attributes.at("POSITION");
    auto accessor = model.accessors[attr];

    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        throw std::exception("POSITION accessor's type or component type is invalid."
                             "the gltf file is corrupted!");
    }

    return load_data<Vec3f>(model, attr);
}

std::vector<Vec3f> load_mesh_normal(const tinygltf::Model &model, const tinygltf::Primitive &primitive) {
    auto attr = primitive.attributes.at("NORMAL");
    auto accessor = model.accessors[attr];

    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {

        throw std::exception("NORMAL accessor's is invalid. the gltf file is corrupted!");
    }

    auto out = load_data<Vec3f>(model, attr);

    return out;
}

std::vector<Vec2f> load_mesh_uvcoord(const tinygltf::Model &model, const tinygltf::Primitive &primitive,
                                     uint32_t uvID) {
    auto attr = primitive.attributes.at("TEXCOORD_" + std::to_string(uvID));
    auto accessor = model.accessors[attr];

    if (accessor.type != TINYGLTF_TYPE_VEC2 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {

        throw std::exception(
            ("TEXCOORD_" + std::to_string(uvID) + " accessor's is invalid. the gltf file is corrupted!")
                .c_str());
    }

    auto out = load_data<Vec2f>(model, attr);

    return out;
}

std::vector<Vec4f> load_mesh_tangent(const tinygltf::Model &model, const tinygltf::Primitive &primitive) {
    auto attr = primitive.attributes.at("TANGENT");
    auto accessor = model.accessors[attr];

    if (accessor.type != TINYGLTF_TYPE_VEC4 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {

        throw std::exception("TANGENT accessor's is invalid. the gltf file is corrupted!");
    }

    auto out = load_data<Vec4f>(model, attr);

    return out;
}

std::vector<Vec4f> load_mesh_color(const tinygltf::Model &model, const tinygltf::Primitive &primitive,
                                   uint32_t color_id) {
    auto attr = primitive.attributes.at("COLOR_" + std::to_string(color_id));
    auto accessor = model.accessors[attr];

    if (accessor.type == TINYGLTF_TYPE_VEC4 && accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return load_data<Vec4f>(model, attr);
    } else if (accessor.type == TINYGLTF_TYPE_VEC3 &&
               accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
        auto out3 = load_data<Vec3f>(model, attr);
        std::vector<Vec4f> out(out3.size());
        for (auto i = 0; i < out.size(); i++) {
            for (int j = 0; j < 3; j++) {
                out[i].data[j][0] = out3[i].data[j][0];
            }
            out[i].data[3][0] = 1;
        }
        return out;
    } else {
        throw std::exception("COLOR_n accessor's is invalid. the gltf file is corrupted!");
    }
}

template <typename DataT>
std::pair<glm::vec3, glm::vec3> get_object_aabb(const std::vector<Vec3<DataT>> &raw_positions) {
    auto positions = convert_container<Vec3f>(raw_positions);
    std::pair<glm::vec3, glm::vec3> out = {};
    if (positions.empty()) {
        return out;
    } else {
        out.first = out.second = {positions[0][0], positions[0][1], positions[0][2]};
    }

    for (size_t i = 1; i < positions.size(); i++) {
        for (int j = 0; j < 3; j++) {
            out.first[j] = std::min(out.first[j], positions[i][j]);
            out.second[j] = std::max(out.second[j], positions[i][j]);
        }
    }
    return out;
}

// NOTE:
template <typename DataT>
std::vector<MeshDesc<DataT>> load_mesh(const tinygltf::Mesh &mesh, const Model &model,
                                       std::vector<RDVertex<float>> &vb, std::vector<uint32_t> &indices,
                                       uint32_t material_offset) {
    // load indices
    std::vector<MeshDesc<DataT>> mesh_out;

    auto meshOutOffset = mesh_out.size();
    mesh_out.resize(mesh_out.size() + mesh.primitives.size());

    for (auto pID = 0; pID < mesh.primitives.size(); pID++) {
        auto &primitive = mesh.primitives[pID];

        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
            throw std::exception("mode of primitive other than TRIANGLES has "
                                 "not been supported yet."
                                 "please triangulate the model using another "
                                 "tool like Blender.");
        }

        auto &out = mesh_out[meshOutOffset + pID];

        // materialID
        out.material_id = primitive.material + material_offset;

        uint32_t offset = (uint32_t)vb.size();

        // indices
        auto new_indices = load_data<rt::Matrix<1, 1, uint32_t>>(model, primitive.indices);
        out.start_index = (uint32_t)indices.size();
        out.n_indices = (uint32_t)new_indices.size();
        std::vector<uint32_t> object_indices;
        for (size_t i = 0; i + 2 < new_indices.size(); i += 3) {
            indices.push_back(offset + new_indices[i + 0].data[0][0]);
            indices.push_back(offset + new_indices[i + 1].data[0][0]);
            indices.push_back(offset + new_indices[i + 2].data[0][0]);

            object_indices.push_back(new_indices[i + 0].data[0][0]);
            object_indices.push_back(new_indices[i + 1].data[0][0]);
            object_indices.push_back(new_indices[i + 2].data[0][0]);
        }

        // vbo
        std::vector<Vec3f> positions, normals, tangents;
        std::vector<Vec2f> uv0, uv1;
        std::vector<Vec4f> color0;

        if (primitive.attributes.count("POSITION")) {
            positions = load_mesh_position(model, primitive);
        } else {
            throw std::exception("POSITION is not set for a mesh");
        }

        // Objects AABB (Top Level Acceleration Structure)
        auto [lower, upper] = get_object_aabb(positions);
        out.aabb_lower = lower;
        out.aabb_upper = upper;

        if (primitive.attributes.count("NORMAL")) {
            // has been normalized..
            normals = load_mesh_normal(model, primitive);
        } else {
            throw std::exception("missing normal");
        }

        if (primitive.attributes.count("TANGENT")) {
            auto tangents4d = load_mesh_tangent(model, primitive);
            for (auto val : tangents4d) {
                // should be normalized...
                Vec3f out;
                for (int i = 0; i < 3; i++) {
                    out.data[i][0] = val.data[i][0];
                }
                out.normalize();
                tangents.emplace_back(out);

                // TODO: handedness
                // out.handedness.emplace_back(val[3] < 0 ? -1 : 1);
            }
        } else {
            RT_WARN("TANGENT is not set for " + mesh.name)
            for (int i = 0; i < normals.size(); i++) {
                // 不支持不设置tangent而设置normalmap
                // 这里设置tangent的主要目的是给采样算法用的，具体数值无所谓，和normal垂直（不平行）就行
                auto a = normals[i][0];
                auto b = normals[i][1];
                auto c = normals[i][2];
                if (abs(a) > 1e-4 || abs(b) > 1e-4) {
                    tangents.emplace_back(Vec3f(-b, a, 0).normalized());
                } else {
                    tangents.emplace_back(Vec3f(0, -c, b).normalized());
                }
            }
        }

        if (primitive.attributes.count("TEXCOORD_0")) {
            // has been normalized..
            uv0 = load_mesh_uvcoord(model, primitive, 0);
        } else {
            for (int i = 0; i < positions.size(); i++) {
                uv0.emplace_back(Vec2f{0, 0});
            }
        }

        if (primitive.attributes.count("TEXCOORD_1")) {
            // has been normalized..
            uv1 = load_mesh_uvcoord(model, primitive, 1);
        } else {
            for (int i = 0; i < positions.size(); i++) {
                uv1.emplace_back(Vec2f{0, 0});
            }
        }

        if (primitive.attributes.count("COLOR_0")) {
            // has been normalized..
            color0 = load_mesh_color(model, primitive, 0);
        } else {
            for (int i = 0; i < positions.size(); i++) {
                color0.emplace_back(Vec4f{1, 1, 1, 1});
            }
        }

        vb.resize(vb.size() + positions.size());

        for (int i = 0; i < positions.size(); i++) {
            auto &vert = vb[offset + i];
#define COPY_MEMORY(dest, src) memcpy(&dest, &src, sizeof(dest))
            COPY_MEMORY(vert.color, color0[i]);
            COPY_MEMORY(vert.normal, normals[i]);
            COPY_MEMORY(vert.position, positions[i]);
            COPY_MEMORY(vert.tangent, tangents[i]);
            COPY_MEMORY(vert.uv0, uv0[i]);
            COPY_MEMORY(vert.uv1, uv1[i]);
#undef COPY_MEMORY
        }

        // BVH (Bottom Level Acceleration Structure) TODO 此处应当用 DataT
        out.object_bvh = std::make_shared<ObjectBVH<DataT>>(positions, object_indices);

        if (mesh.primitives.size() > 100) {
            fprintf(stderr, "%d / %llu primitives is loaded\n", pID, mesh.primitives.size());
        }
    }

    return mesh_out;
}

class CachedTextureLoader {
    std::map<std::pair<int, bool>, std::shared_ptr<RDTexture>> cache;

  public:
    void clearCache() { this->cache.clear(); }
    std::shared_ptr<RDTexture> load_texture(int texture_info_index, bool sRGB, const tinygltf::Model &model);
};

std::shared_ptr<RDTexture> CachedTextureLoader::load_texture(int texture_info_index, bool sRGB,
                                                       const tinygltf::Model &model) {
    if (cache.count({texture_info_index, sRGB}))
        return cache[{texture_info_index, sRGB}];

    if (texture_info_index >= 0) {
        auto textureObj = model.textures[texture_info_index];
        auto textureSource = model.images[textureObj.source];

        int width;
        int height;
        int comp;
        uint8_t *data;
        if (textureSource.bufferView < 0) {
            data = stbi_load(textureSource.uri.c_str(), &width, &height, &comp, STBI_rgb_alpha);
        } else {
            auto buffer_view = model.bufferViews[textureSource.bufferView];
            auto buffer = model.buffers[buffer_view.buffer];
            auto image_data = &buffer.data[buffer_view.byteOffset];
            data = stbi_load_from_memory(image_data, buffer_view.byteLength, &width, &height, &comp,
                                         STBI_rgb_alpha);
        }

        auto out = cache[{texture_info_index, sRGB}] =
            std::make_shared<RDTexture>(data, width, height, sRGB);
        stbi_image_free(data);
        return out;
    }

    return nullptr;
};

template <typename T>
MaterialHolder<T> load_material(const tinygltf::Material &mat, const tinygltf::Model &model,
                                std::map<std::string, uint32_t> textureReg, CachedTextureLoader &loader) {
    MaterialHolder<T> out;
    CUDAMaterial<T> constants;

    constants.color = {(T)mat.pbrMetallicRoughness.baseColorFactor[0],
                       (T)mat.pbrMetallicRoughness.baseColorFactor[1],
                       (T)mat.pbrMetallicRoughness.baseColorFactor[2]};
    constants.emission = {(T)mat.emissiveFactor[0], (T)mat.emissiveFactor[1], (T)mat.emissiveFactor[2]};
    constants.metallic = (T)mat.pbrMetallicRoughness.metallicFactor;
    constants.roughness = (T)mat.pbrMetallicRoughness.roughnessFactor;

    constants.anisotropy = (T)0;

    constants.double_sided = mat.doubleSided;

    out.tex_color = loader.load_texture(mat.pbrMetallicRoughness.baseColorTexture.index, true, model);
    if (out.tex_color) {
        constants.uv_color = mat.pbrMetallicRoughness.baseColorTexture.texCoord;
        constants.tex_color = out.tex_color->tex;
    }

    out.tex_emission = loader.load_texture(mat.emissiveTexture.index, true, model);
    if (out.tex_emission) {
        constants.uv_emission = mat.emissiveTexture.texCoord;
        constants.tex_emission = out.tex_emission->tex;
    }

    auto rm_texture =
        loader.load_texture(mat.pbrMetallicRoughness.metallicRoughnessTexture.index, false, model);
    if (rm_texture) {
        out.tex_metallic = rm_texture;
        constants.uv_metallic = mat.pbrMetallicRoughness.metallicRoughnessTexture.texCoord;
        constants.channel_metallic = 2;

        out.tex_roughness = rm_texture;
        constants.uv_roughness = mat.pbrMetallicRoughness.metallicRoughnessTexture.texCoord;
        constants.channel_roughness = 1;
    }

    // path tracing，不考虑 AO 了
    //    out->texAO = loader.loadTexture(mat.occlusionTexture.index, model);
    //    if (out->texAO) {
    //        constants.uvAO = mat.occlusionTexture.texCoord;
    //        constants.occlusionStrength =
    //        (float)mat.occlusionTexture.strength; constants.channelAO = 0;
    //    }

    out.tex_normal = loader.load_texture(mat.normalTexture.index, false, model);
    if (out.tex_normal)
        constants.uv_normal = mat.normalTexture.texCoord;
    constants.normalmap_scale = (T)mat.normalTexture.scale;

    out.constants = constants;

    return out;
}

template <typename DataT>
std::shared_ptr<hierarchy::Object<DataT>> build_object_hierarchy(
    int in_node_id, const tinygltf::Model &model, std::shared_ptr<hierarchy::Object<DataT>> parent,
    const std::vector<std::vector<std::tuple<int, uint32_t, std::pair<glm::vec3, glm::vec3>, >>> &meshes,
    const std::map<int, rt::hierarchy::Animation> &animations) {
    auto in_node = model.nodes[in_node_id];

    int lightid = -1;

    auto is_ext_light = in_node.extensions.find("KHR_lights_punctual");
    if (is_ext_light != in_node.extensions.end()) {
        lightid = is_ext_light->second.Get("light").GetNumberAsInt();
    }

    bool is_mesh = in_node.mesh >= 0;
    bool is_camera = in_node.camera >= 0;
    bool is_light = lightid >= 0;

    if ((is_mesh ? 1 : 0) + (is_camera ? 1 : 0) + (is_light ? 1 : 0) > 1) {
        throw std::exception("an object can only be one of mesh, camera or light");
    }

    std::shared_ptr<hierarchy::Object<DataT>> out_node;

    if (is_camera) {
        auto &in_camera = model.cameras[in_node.camera];
        auto out_camera = new hierarchy::Camera<DataT>();

        if (in_camera.type == "orthographic") {
            //            outCamera->data.projectionType =
            //            Rendering::RenderingScene::
            //                CameraInfo::ProjectionType::ORTHOGRAPHICS;
            //            outCamera->data.viewWidth =
            //            (float)inCamera.orthographic.xmag * 2;
            //            outCamera->data.viewHeight =
            //            (float)inCamera.orthographic.ymag * 2;
            //            outCamera->data.nearZ =
            //            (float)inCamera.orthographic.znear;
            //            outCamera->data.farZ =
            //            (float)inCamera.orthographic.zfar;
            RT_WARN("orthographic camera is not supported");
        } else if (in_camera.type == "perspective") {
            out_camera->data.aspect_ratio = (DataT)in_camera.perspective.aspectRatio;
            out_camera->data.field_of_view_y = (DataT)in_camera.perspective.yfov;
            out_camera->data.z_near = (DataT)in_camera.perspective.znear;
            out_camera->data.z_far = (DataT)in_camera.perspective.zfar;
        } else {
            throw std::exception("invalid camera type");
        }
        out_node = decltype(out_node)(out_camera);
    } else if (is_light) {
        auto in_light = model.lights[lightid];
        auto out_light = new hierarchy::Light<DataT>();
        // TODO: point and spot lights use luminous intensity in candela (lm/sr)
        // while directional lights use illuminance in lux (lm/m2)
        out_light->data.maximum_distance =
            in_light.range > 0 ? (float)in_light.range : std::numeric_limits<float>::infinity();
        auto intensity = in_light.intensity;
        if (in_light.type == "point") {
            out_light->data.type = RDLight<DataT>::LightType::POINT;
            // TODO...
            // intensity /= 100;
        } else if (in_light.type == "directional") {
            out_light->data.type = RDLight<DataT>::LightType::DIRECTIONAL;
            out_light->data.direction = {0, 0, -1};
        } else if (in_light.type == "spot") {
            out_light->data.type = RDLight<DataT>::LightType::POINT;
            out_light->data.inner_cone_angle = (DataT)in_light.spot.innerConeAngle;
            out_light->data.outer_cone_angle = (DataT)in_light.spot.outerConeAngle;
            out_light->data.direction = {0, 0, -1};
            out_light->data.position = {0, 0, 0};
        } else {
            throw std::exception("unexpected light type");
        }
        out_light->data.intensity = {(float)(in_light.color[0] * intensity),
                                     (float)(in_light.color[1] * intensity),
                                     (float)(in_light.color[2] * intensity)};
        out_node = decltype(out_node)(out_light);
    } else if (is_mesh) {
        out_node = decltype(out_node)(new hierarchy::Object<DataT>());
        for (auto [objid, mat_id, aabb] : meshes[in_node.mesh]) {
            auto mesh_obj = new hierarchy::MeshObject<DataT>();
            //            meshObj->parent = outNode;

            if (mat_id == UINT32_MAX)
                mesh_obj->materialid = 0; // default material id
            else
                mesh_obj->materialid = mat_id;

            mesh_obj->name = in_node.name + " - MESH";

            mesh_obj->objectid = objid;

            mesh_obj->translation = {0, 0, 0};
            mesh_obj->quat_rotation = {1, 0, 0, 0};
            mesh_obj->scaling = {1, 1, 1};

            mesh_obj->aabb_lower_bound = aabb.first;
            mesh_obj->aabb_upper_bound = aabb.second;

            mesh_obj->parent = out_node;

            out_node->children.push_back(std::shared_ptr<hierarchy::Object<DataT>>(mesh_obj));
        }
    } else {
        out_node = decltype(out_node)(new hierarchy::Object<DataT>());
    }

    if (in_node.matrix.empty()) {
        if (!in_node.translation.empty()) {
            out_node->translation = {(float)in_node.translation[0], (float)in_node.translation[1],
                                     (float)in_node.translation[2]};
        }
        if (!in_node.rotation.empty()) {
            out_node->quat_rotation = glm::quat((float)in_node.rotation[3], (float)in_node.rotation[0],
                                                (float)in_node.rotation[1], (float)in_node.rotation[2]);
        }

        if (!in_node.scale.empty()) {
            out_node->scaling = {(float)in_node.scale[0], (float)in_node.scale[1], (float)in_node.scale[2]};
        }
    } else {
        auto &m = in_node.matrix;

        glm::mat4 trans{(float)m[0x0], (float)m[0x1], (float)m[0x2], (float)m[0x3],
                        (float)m[0x4], (float)m[0x5], (float)m[0x6], (float)m[0x7],
                        (float)m[0x8], (float)m[0x9], (float)m[0xA], (float)m[0xB],
                        (float)m[0xC], (float)m[0xD], (float)m[0xE], (float)m[0xF]};

        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(trans, out_node->scaling, out_node->quat_rotation, out_node->translation, skew,
                       perspective);
    }

    if (animations.count(in_node_id)) {
        out_node->animation = animations.at(in_node_id);
    }

    // hierarchy
    out_node->name = in_node.name;
    out_node->parent = parent;
    for (auto child : in_node.children) {
        out_node->children.push_back(build_object_hierarchy(child, model, out_node, meshes, animations));
    }

    return out_node;
}

std::map<int, rt::hierarchy::Animation> load_animations(const tinygltf::Model &model) {
    std::map<int, rt::hierarchy::Animation> animations;

    for (auto animation : model.animations) {
        for (auto channel : animation.channels) {
            if (channel.target_path == "translation") {
                auto times = load_data<Vector<1, float>>(model, animation.samplers[channel.sampler].input);
                auto values = load_data<Vector<3, float>>(model, animation.samplers[channel.sampler].output);
                animations[channel.target_node].translation.times.resize(times.size());
                animations[channel.target_node].translation.values.resize(values.size());
                memcpy(animations[channel.target_node].translation.times.data(), times.data(),
                       sizeof(times[0]) * times.size());
                memcpy(animations[channel.target_node].translation.values.data(), values.data(),
                       sizeof(values[0]) * values.size());
            } else if (channel.target_path == "scale") {
                auto times = load_data<Vector<1, float>>(model, animation.samplers[channel.sampler].input);
                auto values = load_data<Vector<3, float>>(model, animation.samplers[channel.sampler].output);
                animations[channel.target_node].scale.times.resize(times.size());
                animations[channel.target_node].scale.values.resize(values.size());
                memcpy(animations[channel.target_node].scale.times.data(), times.data(),
                       sizeof(times[0]) * times.size());
                memcpy(animations[channel.target_node].scale.values.data(), values.data(),
                       sizeof(values[0]) * values.size());
            } else if (channel.target_path == "rotation") {
                auto times = load_data<Vector<1, float>>(model, animation.samplers[channel.sampler].input);
                auto values = load_data<Vector<4, float>>(model, animation.samplers[channel.sampler].output);
                animations[channel.target_node].rotation.times.resize(times.size());
                animations[channel.target_node].rotation.values.reserve(values.size());
                memcpy(animations[channel.target_node].rotation.times.data(), times.data(),
                       sizeof(times[0]) * times.size());
                for (auto value : values) {
                    animations[channel.target_node].rotation.values.push_back(
                        glm::quat(value[3], value[0], value[1], value[2]));
                }
            } else {
                RT_WARN("unsupported path: " + channel.target_path);
            }
        }
    }

    return animations;
}

// RDResource 中涉及 GPU 资源（VBO），所以不那么容易转换
template <typename DataT>
std::shared_ptr<hierarchy::Object<DataT>> load_gltf2(const std::string &path,
                                                     RDResource<DataT> &rd_resource) {
    auto dot_position = path.rfind('.');
    std::string extension;
    if (dot_position != std::string::npos) {
        extension = path.substr(dot_position);
    }

    Model model;
    TinyGLTF loader;
    std::string err;
    std::string warn;

    bool load_succeeded = false;

    if (extension == ".glb" || extension == ".GLB") {
        load_succeeded = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else if (extension == ".gltf" || extension == ".GLTF") {
        load_succeeded = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    } else {
        throw std::exception("the extension of glTF2 file(`%s`) should be .glb or .gltf");
    }

    if (!warn.empty()) {
        RT_WARN(warn);
    }

    if (!load_succeeded) {
        throw std::exception(err.c_str());
    }

    std::map<std::string, std::uint32_t> texture_reg;
    std::vector<RDVertex<float>> vbo_data;
    std::vector<std::vector<MeshDesc<DataT>>> mesh_in;
    std::vector<uint32_t> indices;

    // default material
    if (rd_resource.materials.empty()) {
        rd_resource.materials.push_back(MaterialHolder<DataT>());
    }

    int i = 0;
    for (auto mesh : model.meshes) {
        mesh_in.push_back(load_mesh<DataT>(mesh, model, vbo_data, indices, rd_resource.materials.size()));
        if (model.meshes.size() > 100) {
            fprintf(stderr, "%d / %llu mesh is loaded\n", ++i, model.meshes.size());
        }
    }

     rd_resource.compute_m(vbo_data, indices);

    std::vector<std::vector<std::tuple<int, uint32_t, std::pair<glm::vec3, glm::vec3>>>> meshes;
    std::vector<std::shared_ptr<GLMeshVAO>> mesh_vaos;

    CachedTextureLoader texture_loader;

    for (auto &material : model.materials) {
        rd_resource.materials.push_back(load_material<DataT>(material, model, texture_reg, texture_loader));
    }

    uint32_t vbo_index = rd_resource.objects_vbo.size();
    uint32_t ebo_index = rd_resource.objects_ebo.size();
    std::shared_ptr<MeshEBO> ebo(new MeshEBO(indices));
    rd_resource.objects_ebo.push_back(ebo);
    std::shared_ptr<MeshVBO<DataT>> vbo(new MeshVBO<DataT>(convert_container<RDVertex<DataT>>(vbo_data)));
    rd_resource.objects_vbo.push_back(vbo);

    for (auto mesh_group : mesh_in) {
        meshes.push_back({});
        for (const MeshDesc<DataT> &mesh : mesh_group) {
            rd_resource.objects_vao.push_back(std::shared_ptr<GLMeshVAO>(
                new GLMeshVAO(vbo_index, ebo_index, mesh.start_index, mesh.n_indices,
                              rd_resource.materials[mesh.material_id].constants.double_sided, rd_resource)));
//            rd_resource.objects_aabb.push_back({mesh.aabb_lower, mesh.aabb_upper});
            rd_resource.objects_bvh.push_back(mesh.object_bvh);

            meshes.rbegin()->push_back({static_cast<int>(rd_resource.objects_vao.size()) - 1,
                                        mesh.material_id,
                                        {mesh.aabb_lower, mesh.aabb_upper}});
        }
    }

    auto animations = load_animations(model);

    std::shared_ptr<hierarchy::Object<DataT>> root_object(new hierarchy::Object<DataT>());
    for (auto nodeID : model.scenes[model.defaultScene].nodes) {
        root_object->children.push_back(
            build_object_hierarchy(nodeID, model, root_object, meshes, animations));
    }

    return root_object;
}

template std::shared_ptr<hierarchy::Object<float>> load_gltf2<float>(const std::string &path,
                                                                     RDResource<float> &rd_resource);
template std::shared_ptr<hierarchy::Object<float16>> load_gltf2<float16>(const std::string &path,
                                                                         RDResource<float16> &rd_resource);

} // namespace rt
