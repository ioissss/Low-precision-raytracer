#ifndef RT_TRACE_OBJECT_BVH_HPP
#define RT_TRACE_OBJECT_BVH_HPP

#include "math/matrix.hpp"

#include <algorithm>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

namespace rt {


template <typename DataT> struct ObjectBVH {
    struct AABB {
        Vec3<DataT> lower_bound = {};
        Vec3<DataT> upper_bound = {};
    };

    // ROOT 位于下标 0 的位置, UINT32_MAX 表示无孩子节点
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

        // 为 stackless 遍历做准备
        uint32_t parent;

        enum class NodeType { BVH_CHILDREN, GEOMETRY_CHILDREN } type;

        AABB aabb;

        __device__ bool is_leaf() const { return type == NodeType::GEOMETRY_CHILDREN; }
    };

    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<uint32_t> geometry_offset; // 顶点偏置

    ObjectBVH() = default;

    // 构造 BVH 的时候用 f32
    ObjectBVH(const std::vector<Vec3f> &positions, const std::vector<uint32_t> &indices) {
        std::vector<uint32_t> geometries(indices.size() / 3);
        for (size_t i = 0; i < geometries.size(); i++) {
            // 三角形在 EBO 中的起始下标编号
            geometries[i] = i * 3;
        }

        thrust::host_vector<BVHNode> out;
        thrust::host_vector<uint32_t> geo_offsets;
        build_object_bvh(geometries.begin(), geometries.end(), out, geo_offsets, positions, indices,
                         UINT32_MAX);

        this->nodes = out;
        this->geometry_offset = geo_offsets;
    }

    struct Ref {
        BVHNode *nodes = nullptr;
        uint32_t *geometry_offset = nullptr;
    };

    Ref get_ref() {
        return {thrust::raw_pointer_cast(nodes.data()), thrust::raw_pointer_cast(geometry_offset.data())};
    }

  private:
    constexpr static uint32_t MAX_GEOMETRY_PER_LEAF = 1;
    uint32_t build_object_bvh(std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end,
                              thrust::host_vector<BVHNode> &out, thrust::host_vector<uint32_t> &geo_offsets,
                              const std::vector<Vec3f> &positions, const std::vector<uint32_t> &indices,
                              uint32_t parent) {
        size_t size = std::distance(begin, end);
        if (size == 0)
            return UINT32_MAX;

        // 确定沿着哪个轴进行划分
        Vec3f min_value;
        Vec3f max_value;
        min_value = max_value = positions[indices[*begin]];

        for (auto it = begin; it < end; it++) {
            for (int j = 0; j < 3; j++) {
                min_value = elemin(min_value, positions[indices[j + *it]]);
                max_value = elemax(max_value, positions[indices[j + *it]]);
            }
        }

        out.push_back({});
        auto index = out.size() - 1;
        out[index].parent = parent;
        out[index].aabb.lower_bound = min_value.as<DataT>();
        out[index].aabb.upper_bound = max_value.as<DataT>();

        if (size <= MAX_GEOMETRY_PER_LEAF) {
            auto offset = geo_offsets.size();
            for (auto it = begin; it < end; ++it) {
                geo_offsets.push_back(*it);
            }
            out[index].geometry_children.offset = offset;
            out[index].geometry_children.length = size;
            out[index].type = BVHNode::NodeType::GEOMETRY_CHILDREN;
        } else {
            Vec3f width = max_value - min_value;
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
                             [split_axis, &positions, &indices](uint32_t &x, uint32_t &y) {
                return positions[indices[x]][split_axis] < positions[indices[y]][split_axis];
            });

            out[index].bvh_children.lc =
                build_object_bvh(begin, begin + split_pos, out, geo_offsets, positions, indices, index);
            out[index].bvh_children.rc =
                build_object_bvh(begin + split_pos, end, out, geo_offsets, positions, indices, index);
            out[index].type = BVHNode::NodeType::BVH_CHILDREN;
        }
        return index;
    }
};

} // namespace rt

#endif // !RT_TRACE_OBJECT_BVH_HPP
