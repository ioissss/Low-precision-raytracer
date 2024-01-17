// Ϊ������������ object ��ε� bvh
// �� BVH ��Ҫÿ֡���й��������ٶȵ�Ҫ���

#ifndef RT_TRACE_SCENE_BVH_HPP
#define RT_TRACE_SCENE_BVH_HPP

#include "math/matrix.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

namespace rt {

template <typename DataT> struct SceneBVH {
    struct AABB {
        Vec3<DataT> lower_bound = {};
        Vec3<DataT> upper_bound = {};
    };

    // ROOT λ���±� 0 ��λ��, UINT32_MAX ��ʾ�޺��ӽڵ�
    struct BVHNode {
        union {
            struct {
                uint32_t lc;
                uint32_t rc;
            } bvh_children;

            struct {
                uint32_t offset;
                uint32_t length;
            } geometry_children;
        };

        // Ϊ stackless ������׼��
        uint32_t parent;

        enum class NodeType { BVH_CHILDREN, GEOMETRY_CHILDREN } type;

        __device__ bool is_leaf() const { return type == NodeType::GEOMETRY_CHILDREN; }

        AABB aabb;
    };

    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<uint32_t> inscene_objid; // VAO ���±� Index
    thrust::device_vector<AABB> aabbs_W;  // �� `inscene_objid` ˳��һ�µ� aabbs�����������ʾ

    SceneBVH() = default;
    void update(const std::vector<std::pair<glm::vec3, glm::vec3>> &aabbs,
                const std::vector<int> vao_indices) {
        std::vector<uint32_t> aabb_enum(aabbs.size());
        for (size_t i = 0; i < aabbs.size(); i++) {
            aabb_enum[i] = i;
        }

        thrust::host_vector<BVHNode> out;
        thrust::host_vector<uint32_t> vaos;
        build_scene_bvh(aabbs, aabb_enum.begin(), aabb_enum.end(), out, vaos, vao_indices, UINT32_MAX);
        thrust::host_vector<AABB> host_aabbs(aabb_enum.size());
        for (int i = 0; i < aabb_enum.size(); ++i) {
            auto id = aabb_enum[i];
            host_aabbs[i].lower_bound = glm_to_matrix<DataT>(aabbs[id].first);
            host_aabbs[i].upper_bound = glm_to_matrix<DataT>(aabbs[id].second);
        }

        this->nodes = out;
        this->inscene_objid = vaos;
        this->aabbs_W = host_aabbs;
    }

  private:
    constexpr static uint32_t MAX_GEOMETRY_PER_LEAF = 1;
    uint32_t build_scene_bvh(const std::vector<std::pair<glm::vec3, glm::vec3>> &aabbs,
                             std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end,
                             thrust::host_vector<BVHNode> &out, thrust::host_vector<uint32_t> &vaos,
                             const std::vector<int> vao_indices, uint32_t parent) {
        size_t size = std::distance(begin, end);
        if (size == 0)
            return UINT32_MAX;

        // ȷ�������ĸ�����л���
        glm::vec3 min_value;
        glm::vec3 max_value;
        min_value = aabbs[*begin].first;
        max_value = aabbs[*begin].second;

        for (auto it = begin + 1; it < end; it++) {
            min_value = glm::min(min_value, aabbs[*it].first);
            max_value = glm::max(max_value, aabbs[*it].second);
        }

        out.push_back({});
        auto index = out.size() - 1;
        out[index].parent = parent;
        out[index].aabb.lower_bound = glm_to_matrix<DataT>(min_value);
        out[index].aabb.upper_bound = glm_to_matrix<DataT>(max_value);

        if (size <= MAX_GEOMETRY_PER_LEAF) {
            out[index].type = BVHNode::NodeType::GEOMETRY_CHILDREN;
            auto vao_offset = vaos.size();
            for (auto it = begin; it < end; it++) {
                vaos.push_back(*it);
            }
            out[index].geometry_children.offset = vao_offset;
            out[index].geometry_children.length = size;
        } else {

            glm::vec3 width = max_value - min_value;
            size_t split_axis = 0;
            if (width[1] > width[0] && width[1] > width[2]) {
                split_axis = 1;
            } else if (width[2] > width[0] && width[2] > width[1]) {
                split_axis = 2;
            } else {
                split_axis = 0;
            }

            size_t split_pos = size / 2;
            std::nth_element(begin, begin + split_pos, end,
                             [split_axis, &aabbs](uint32_t &x_idx, uint32_t &y_idx) {
                                 return aabbs[x_idx].first[split_axis] < aabbs[y_idx].first[split_axis];
                             });

            out[index].bvh_children.lc =
                build_scene_bvh(aabbs, begin, begin + split_pos, out, vaos, vao_indices, index);
            out[index].bvh_children.rc =
                build_scene_bvh(aabbs, begin + split_pos, end, out, vaos, vao_indices, index);
            out[index].type = BVHNode::NodeType::BVH_CHILDREN;
        }

        return index;
    }
};

} // namespace rt

#endif // !RT_TRACE_SCENE_BVH_HPP
