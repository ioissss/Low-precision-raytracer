/// 该文件表示场景层次关系，需要支持泛型

#ifndef RT_RTRT_HIERARCHY_HPP
#define RT_RTRT_HIERARCHY_HPP

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/wrap.hpp"
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include "math/matrix.hpp"
#include "memory.hpp"

namespace rt {
namespace hierarchy {

static glm::vec3 apply_to_position(const glm::mat4 &mat, const glm::vec3 &position) {
    glm::vec4 pos4(position.x, position.y, position.z, 1);
    auto out4 = mat * pos4;
    return glm::vec3(out4.x / out4.w, out4.y / out4.w, out4.z / out4.w);
}

static void swap_to_minmax(glm::vec3 &a, glm::vec3 &b) {
    for (int i = 0; i < 3; i++) {
        if (a[i] > b[i]) {
            std::swap(a[i], b[i]);
        }
    }
}

template <typename DataT> struct Object;

template <typename DataT> struct MeshObject : public Object<DataT> {
    // 必须设置一个材料。不同的 vbo 结构不一样，做不了默认材料
    uint32_t materialid;
    uint32_t objectid;

    glm::vec3 aabb_lower_bound;
    glm::vec3 aabb_upper_bound;

    std::pair<glm::vec3, glm::vec3> get_aabb(glm::mat4 transform) {
        glm::vec3 bounds[2] = {aabb_lower_bound, aabb_upper_bound};
        glm::vec3 out_lower, out_upper;
        for (int i = 0; i < 8; i++) {
            auto vec4 = glm::vec4{bounds[(i & 1) >> 0][0], bounds[(i & 2) >> 1][1], bounds[(i & 4) >> 2][2], 1};
            vec4 = transform * vec4;
            auto vec3 = glm::vec3{vec4.x / vec4.w, vec4.y / vec4.w, vec4.z / vec4.w};
            for (int j = 0; j < 3; j++) {
                if (i == 0 || vec3[j] > out_upper[j]) {
                    out_upper[j] = vec3[j];
                }
                if (i == 0 || vec3[j] < out_lower[j]) {
                    out_lower[j] = vec3[j];
                }
            };
        }
        
        return {out_lower, out_upper};
    }
};

template <typename DataT> struct Camera : public Object<DataT> {

    RDCamera<DataT> data;

    glm::mat4 world_to_view(glm::mat4 transform_mat) {
        return glm::lookAt(glm::vec3(transform_mat * glm::vec4(0, 0, 0, 1)),
                           glm::vec3(transform_mat * glm::vec4(0, 0, -1, 1)),
                           glm::vec3(transform_mat * glm::vec4(0, 1, 0, 0)));
    }
};

template <typename DataT> struct Light : public Object<DataT> {
    Light() = default;

    RDLight<DataT> data;
};

template <typename T> struct Sampler {
    std::vector<float> times;
    std::vector<T> values;
    size_t cursor = 0;

    T sample(float time, bool mod, T default_value) {
        if (times.size() == 1) {
            return values[0];
        }

        if (times.empty()) {
            return default_value;
        }

        float max_time = *times.rbegin();
        if (time >= max_time) {
            if (!mod)
                return *values.rbegin();
            time = fmodf(time, *times.rbegin());
            cursor = 0;
        }

        while (times[cursor + 1] < time) {
            cursor++;
        }

        auto u = (time - times[cursor]) / (times[cursor + 1] - times[cursor]);
        return interpolate(values[cursor], values[cursor + 1], u);
    }

    static glm::vec3 interpolate(glm::vec3 a, glm::vec3 b, float w) { return (1 - w) * a + w * b; }
    static glm::quat interpolate(glm::quat a, glm::quat b, float w) { return glm::lerp(a, b, w); }
};

struct Animation {
    Sampler<glm::vec3> translation;
    Sampler<glm::vec3> scale;
    Sampler<glm::quat> rotation;
};

template <typename DataT> struct Object {
    std::string name;

    glm::vec3 translation{0, 0, 0};
    glm::quat quat_rotation{1, 0, 0, 0};
    glm::vec3 scaling{1, 1, 1};

    Animation animation;

    std::vector<std::shared_ptr<Object>> children;
    std::weak_ptr<Object> parent;

    Object() = default;

    void apply_animation(float time) {
        translation = animation.translation.sample(time, true, translation);
        scaling = animation.scale.sample(time, true, scaling);
        quat_rotation = animation.rotation.sample(time, true, quat_rotation);
            
        for (auto child: children) {
            child->apply_animation(time);
        }
    }

    glm::mat4 transform_matrix() const {
        return glm::translate(glm::identity<glm::mat4>(), translation) *
               glm::rotate(glm::identity<glm::mat4>(), glm::angle(quat_rotation), glm::axis(quat_rotation)) *
               glm::scale(glm::identity<glm::mat4>(), scaling);
    }

    void set_transform_matrix(const glm::mat4& value) {
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(value, this->scaling, this->quat_rotation, this->translation, skew,
                       perspective);
        // this->quat_rotation = glm::conjugate(this->quat_rotation);
    }

    glm::mat4 local_to_world() const {
        auto pparent = parent.lock();
        if (pparent == nullptr) {
            return transform_matrix();
        } else {
            return pparent->local_to_world() * transform_matrix();
        }
    }

    virtual ~Object() = default;

    std::shared_ptr<Object<DataT>> insert_parent(std::shared_ptr<Object<DataT>> self) {
        auto parent = this->parent.lock();
        auto insert_obj = std::make_shared<Object<DataT>>();
        self->parent = insert_obj;
        insert_obj->children = {self};
        insert_obj->translation = self->translation;
        insert_obj->quat_rotation = self->quat_rotation;
        insert_obj->scaling = self->scaling;

        self->translation = glm::vec3{0, 0, 0};
        self->quat_rotation = glm::quat{1, 0, 0, 0};
        self->scaling = glm::vec3{1, 1, 1};

        if (parent != nullptr) {
            for (auto& child: parent->children) {
                if (&*child == this) {
                    child = insert_obj;
                }
            }
        }

        return insert_obj;
        
    }

    virtual std::shared_ptr<Object<DataT>> search_children(const std::string &name) {
        for (auto &child : children) {
            if (child->name == name) {
                return child;
            }
            auto rst = child->search_children(name);
            if (rst)
                return rst;
        }
        return nullptr;
    }

    virtual std::shared_ptr<Camera<DataT>> search_camera() {
        for (auto &child : children) {
            if (auto camera = std::dynamic_pointer_cast<Camera<DataT>>(child); camera) {
                return camera;
            }
            auto rst = child->search_camera();
            if (rst)
                return rst;
        }
        return nullptr;
    }
};

// 该数据结构记录一种层次关系
// 里面的信息均以及转移到 GPU，只要做简单组装即可
template <typename DataT> struct Scene {
    static Mat4<DataT> convert_matrix(glm::mat4 data) {
        Mat4f temp;
        memcpy(temp.data, &data, sizeof(data));
        return temp.as<DataT>();
    }

  public:
    std::shared_ptr<Object<DataT>> root_object;
    std::shared_ptr<Camera<DataT>> active_camera;

  protected:
    void build_rendering_scene_recursively(std::shared_ptr<RDScene<DataT>> dest,
                                           std::shared_ptr<Object<DataT>> node, glm::mat4 transform) {
        if (node == nullptr)
            return;

        // DirectX: 行主序
        glm::mat4 new_transform = transform * node->transform_matrix();

        if (auto mesh = std::dynamic_pointer_cast<MeshObject<DataT>>(node); mesh) {
            dest->objects_constants.push_back(
                {new_transform, glm::inverse(new_transform), mesh->materialid, mesh->objectid});
            dest->objects_aabb.push_back(mesh->get_aabb(new_transform));
        } else if (auto light = std::dynamic_pointer_cast<Light<DataT>>(node); light) {
            RDLight<DataT> ldesc = light->data;

            // normal
            glm::vec4 dir_w = glm::normalize((new_transform * glm::vec4(0, 0, -1, 0)));
            for (auto i = 0; i < 3; i++) {
                ldesc.direction[i] = dir_w[i];
                ldesc.position[i] = new_transform[3][i];
            }

            dest->lights.push_back(ldesc);
        } else if (auto camera = std::dynamic_pointer_cast<Camera<DataT>>(node);
                   camera && camera == active_camera) {
            // dest->camera = camera->data;
            // dest->camera.trans_W2V = camera->getW2VMatrix(newTransform);
            dest->camera = camera->data;
            Mat4f w2v;
            auto glm_w2v = camera->world_to_view(new_transform);
            glm_w2v = glm::transpose(glm_w2v);
            memcpy(w2v.data, glm::value_ptr(glm_w2v), sizeof(glm_w2v));
            dest->camera.transform = Mat4<DataT>(w2v);
            
            dest->camera.transform_L2W = convert_matrix(glm::transpose(new_transform));
        }

        for (auto child : node->children) {
            build_rendering_scene_recursively(dest, child, new_transform);
        }
    }

  public:
    RDCamera<DataT> camera_info(std::shared_ptr<Camera<DataT>> camera) {
        auto out = camera->data;
        out.trans_W2V = camera->world_to_view(camera->getLocalToWorldMatrix());
        return out;
    }

    std::shared_ptr<RDScene<DataT>> build_rendering_scene() {
        if (this->active_camera == nullptr) {
            throw std::exception("no active camera");
        }
        std::shared_ptr<RDScene<DataT>> out(new RDScene<DataT>());
        build_rendering_scene_recursively(out, root_object, glm::identity<glm::mat4>());
        return out;
    }

    std::shared_ptr<Object<DataT>> search_object(const std::string &name) {
        // 以后可以做点 cache，或者做点预处理
        if (root_object->name == name) {
            return root_object;
        }
        return root_object->search_children(name);
    }
    std::shared_ptr<Camera<DataT>> search_camera() {
        // 以后可以做点 cache，或者做点预处理
        if (auto camera = std::dynamic_pointer_cast<Camera<DataT>>(root_object); camera) {
            return camera;
        }
        return root_object->search_camera();
    }
};

} // namespace hierarchy
} // namespace rt

#endif // !RT_RTRT_HIERARCHY_HPP
