#ifndef RTRT_CUDA_HPP
#define RTRT_CUDA_HPP

#include "bsdf.hpp"
#include "math/matrix.hpp"
#include "math/number.hpp"
#include "memory.hpp"
#include "trace/scene_bvh.hpp"
// #include "numbers/all.hpp"
// #include "utils/config.hpp"
#include "gui/imgui_window.hpp"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <nvfunctional>
#include <surface_functions.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace rt {

__constant__ DebugInfo dev_debug_info;

#define SKYBOX_COLOR (RGBColor<DataT>(0.00, 0.00, 0.00))

__device__ bool debug_thread() {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    int img_x = thread_id % dev_debug_info.window_width;
    int img_y = thread_id / dev_debug_info.window_width;

    return img_x == dev_debug_info.window_x && img_y == dev_debug_info.window_y;
}

template <typename DataT> struct Ray {
    Vec3<DataT> source;
    Vec3<DataT> direction;

    __device__ Ray<DataT> transformed(const Mat4<DataT> &transform) const {
        auto out_src4 = transform * Vec4<DataT>(source[0], source[1], source[2], 1);
        auto out_dir3 = (transform * Vec4<DataT>(direction[0], direction[1], direction[2], 0)).xyz().clone();
        return {Vec3<DataT>(out_src4[0] / out_src4[3], out_src4[1] / out_src4[3], out_src4[2] / out_src4[3]),
                out_dir3};
    }
};

template <typename DataT> struct Intersection {
    uint32_t inscene_objid;
    uint32_t inobject_offset;
    Vec3<DataT> barycentric;
    DataT t = 1e5;
    enum State { INTERSECTED_FRONT, INTERSECTED_BACK, NONE } state = NONE;
    __device__ bool intersected() const { return state != NONE; }
};

template <typename DataT> __device__ void swap(DataT &a, DataT &b) noexcept {
    auto c = a;
    a = b;
    b = c;
}

template <typename DataT> __device__ void swap_to_ordered(DataT &a, DataT &b) {
    if (b < a)
        swap(a, b);
}

template <typename DataT>
__device__ bool ray_aabb_intersection_scene(const Ray<DataT> &ray, const typename SceneBVH<DataT>::AABB aabb,
                                            DataT *min_t, DataT *max_t) {

    auto t1 = elediv(aabb.lower_bound - ray.source, ray.direction);
    auto t2 = elediv(aabb.upper_bound - ray.source, ray.direction);

    for (int i = 0; i < 3; i++)
        swap_to_ordered(t1[i], t2[i]);

    bool updated = false;
    DataT t1max = 0;
    DataT t2min = 0;

    for (int i = 0; i < 3; i++) {
        if (!(isfinite(t1[i]) && isfinite(t2[i])))
            continue;
        if (!updated || (t1max < t1[i]))
            t1max = t1[i];
        if (!updated || (t2[i] < t2min))
            t2min = t2[i];
        updated = true;
    }

    if (!updated) {
        return false;
    }

    bool out = t1max <= t2min + DataT(0.02f) && /*min_t*/ 0 <= t2min + DataT(0.02f);

    if (out) {
        if (min_t)
            *min_t = t1max;
        if (max_t)
            *max_t = t2min;
    }

    return out;
}

template <typename DataT>
__device__ bool ray_aabb_intersection_object(const Ray<DataT> &ray,
                                             const typename ObjectBVH<DataT>::AABB &aabb, DataT *min_t,
                                             DataT *max_t) {

    auto t1 = elediv(aabb.lower_bound - ray.source, ray.direction);
    auto t2 = elediv(aabb.upper_bound - ray.source, ray.direction);

    for (int i = 0; i < 3; i++)
        swap_to_ordered(t1[i], t2[i]);

    bool updated = false;
    DataT t1max = 0;
    DataT t2min = 0;

    for (int i = 0; i < 3; i++) {
        if (!(isfinite(t1[i]) && isfinite(t2[i])))
            continue;
        if (!updated || (t1max < t1[i]))
            t1max = t1[i];
        if (!updated || (t2[i] < t2min))
            t2min = t2[i];
        updated = true;
    }

    if (!updated) {
        return false;
    }

    bool out = t1max <= t2min * DataT(1.001953f) && /*min_t*/ 0 <= t2min;

    if (out) {
        if (min_t)
            *min_t = t1max;
        if (max_t)
            *max_t = t2min;
    }

    return out;
}

template <typename DataT> struct CUDARenderGIInput {
    // G-Buffer
    cudaSurfaceObject_t gbuffer_color;
    cudaSurfaceObject_t gbuffer_inscene_objid;
    cudaSurfaceObject_t gbuffer_normal_depth;
    cudaSurfaceObject_t gbuffer_position;
    cudaSurfaceObject_t gbuffer_tangent;
    cudaSurfaceObject_t gbuffer_uv0uv1;
    cudaSurfaceObject_t gbuffer_inobject_offset;

    // EBO & VBO
    RDVertex<DataT> const *const *vbos = nullptr;
    uint32_t const *const *ebos = nullptr;

    // M_shift
    Matrix<3, 3, DataT> *M_shift = nullptr;
    Matrix<3, 3, float> *M_shift_f32 = nullptr;
    Vec3<float> *position_f32 = nullptr;

    // 若干个 VAO
    const CUDAMeshVAO *vaos = nullptr;
    // 总的 vao 个数
    uint32_t n_vaos = 0;

    // Material Constant
    const CUDAMaterial<DataT> *material_constants = nullptr;

    RDLight<DataT> *lights = nullptr;
    uint32_t n_lights;

    CUDAObjectConstantData<DataT> *objects_constant = nullptr;

    typename ObjectBVH<DataT>::Ref *objects_bvh = nullptr;

    typename RDSkybox<DataT>::Ref skybox;

    struct SkipGeometry {
        uint32_t vao_index = UINT32_MAX;
        uint32_t inobject_offset;
    };

    // 返回值是是否相交，和 intersection 是否使得 t 变小无关
    template <bool culling>
    __device__ bool ray_triangle_intersection(const Ray<DataT> &ray, uint32_t inscene_objid,
                                              uint32_t vao_index, uint32_t inobject_offset,
                                              Intersection<DataT> &intersection, DataT min_distance,
                                              DataT max_distance, DataT *out_t) const {

        auto vao = vaos[vao_index];
        auto offset = inobject_offset + vao.ebo_offset;
        // auto v0 = vbos[vao.vbo_index][ebos[vao.ebo_index][offset + 0]].position;
        // auto v1 = vbos[vao.vbo_index][ebos[vao.ebo_index][offset + 1]].position;
        auto v2 = vbos[vao.vbo_index][ebos[vao.ebo_index][offset + 2]].position;

        auto O = ray.source;
        auto D = ray.direction;
        auto m0 = M_shift[offset / 3].row(0);
        auto m1 = M_shift[offset / 3].row(1);
        auto m2 = M_shift[offset / 3].row(2);

        O[0] = O[0] - v2[0];
        O[1] = O[1] - v2[1];
        O[2] = O[2] - v2[2];

        auto Ox_0 = O[0] * m0[{0, 0}];
        auto Ox_1 = O[1] * m0[{0, 1}];
        auto Ox_2 = O[2] * m0[{0, 2}];
        auto Dx_0 = D[0] * m0[{0, 0}];
        auto Dx_1 = D[1] * m0[{0, 1}];
        auto Dx_2 = D[2] * m0[{0, 2}];
        auto Oy_0 = O[0] * m1[{0, 0}];
        auto Oy_1 = O[1] * m1[{0, 1}];
        auto Oy_2 = O[2] * m1[{0, 2}];
        auto Dy_0 = D[0] * m1[{0, 0}];
        auto Dy_1 = D[1] * m1[{0, 1}];
        auto Dy_2 = D[2] * m1[{0, 2}];

        DataT Ox = Ox_0 + Ox_1 + Ox_2;
        DataT Dx = Dx_0 + Dx_1 + Dx_2;
        DataT Oy = Oy_0 + Oy_1 + Oy_2;
        DataT Dy = Dy_0 + Dy_1 + Dy_2;
        /*DataT Oz = O[0] * m2[{0, 0}] + O[1] * m2[{0, 1}] + O[2] * m2[{0, 2}];
        DataT Dz = D[0] * m2[{0, 0}] + D[1] * m2[{0, 1}] + D[2] * m2[{0, 2}];*/
        float Oz = float(O[0]) * float(m2[{0, 0}]) + float(O[1]) * float(m2[{0, 1}]) +
                   float(O[2]) * float(m2[{0, 2}]);
        float Dz = float(D[0]) * float(m2[{0, 0}]) + float(D[1]) * float(m2[{0, 1}]) +
                   float(D[2]) * float(m2[{0, 2}]);

        float invDz = 1.0 / Dz;
        float t = -Oz * invDz;
        DataT t_Dx = t * Dx;
        DataT t_Dy = t * Dy;
        DataT u = Ox + t_Dx;
        DataT v = Oy + t_Dy;

        //计算误差
        DataT delta1 = powf(2, -10);
        DataT delta2 = powf(2, -8);
        DataT error_Ox =
            delta1 * (abs(Ox_0) + abs(Ox_1) + abs(Ox_2)) + delta2 * (abs(Ox_0) + abs(Ox_1) + abs(Ox_2));
        DataT error_Dx =
            delta1 * (abs(Dx_0) + abs(Dx_1) + abs(Dx_2)) + delta2 * (abs(Dx_0) + abs(Dx_1) + abs(Dx_2));
        DataT error_Oy =
            delta1 * (abs(Oy_0) + abs(Oy_1) + abs(Oy_2)) + delta2 * (abs(Oy_0) + abs(Oy_1) + abs(Oy_2));
        DataT error_Dy =
            delta1 * (abs(Dy_0) + abs(Dy_1) + abs(Dy_2)) + delta2 * (abs(Dy_0) + abs(Dy_1) + abs(Dy_2));

        DataT error_u =
            (error_Ox + DataT(t) * error_Dx + delta1 * (abs(Ox) + DataT(3) * abs(t_Dx))) * DataT(0.2);
        DataT error_v =
            (error_Oy + DataT(t) * error_Dy + delta1 * (abs(Oy) + DataT(3) * abs(t_Dy))) * DataT(0.2);

        if (t > min_distance && t < intersection.t && t < max_distance) {

            // float32运算
            auto w = 1 - u - v;
            if ((u >= DataT(-error_u) && u <= DataT(0.0)) || (v >= DataT(-error_v) && v <= DataT(0.0)) ||
                (w >= DataT(-(error_v + error_u)) && w <= DataT(0.0))) {

                //光线生成就不转了
                Matrix<3, 3, float> M2_f32 = M_shift_f32[offset / 3];
                auto m0_f32 = M2_f32.row(0);
                auto m1_f32 = M2_f32.row(1);
                auto m2_f32 = M2_f32.row(2);

                Vec3<float> v2_f32 = position_f32[ebos[vao.ebo_index][offset + 2]];
                Vec3<float> O_f32;
                O_f32[0] = float(ray.source[0]) - v2_f32[0];
                O_f32[1] = float(ray.source[1]) - v2_f32[1];
                O_f32[2] = float(ray.source[2]) - v2_f32[2];

                Vec3<float> D_f32;
                D_f32[0] = float(ray.direction[0]);
                D_f32[1] = float(ray.direction[1]);
                D_f32[2] = float(ray.direction[2]);

                float Ox_f32 =
                    O_f32[0] * m0_f32[{0, 0}] + O_f32[1] * m0_f32[{0, 1}] + O_f32[2] * m0_f32[{0, 2}];
                float Dx_f32 =
                    D_f32[0] * m0_f32[{0, 0}] + D_f32[1] * m0_f32[{0, 1}] + D_f32[2] * m0_f32[{0, 2}];
                float Oy_f32 =
                    O_f32[0] * m1_f32[{0, 0}] + O_f32[1] * m1_f32[{0, 1}] + O_f32[2] * m1_f32[{0, 2}];
                float Dy_f32 =
                    D_f32[0] * m1_f32[{0, 0}] + D_f32[1] * m1_f32[{0, 1}] + D_f32[2] * m1_f32[{0, 2}];
                float Oz_f32 =
                    O_f32[0] * m2_f32[{0, 0}] + O_f32[1] * m2_f32[{0, 1}] + O_f32[2] * m2_f32[{0, 2}];
                float Dz_f32 =
                    D_f32[0] * m2_f32[{0, 0}] + D_f32[1] * m2_f32[{0, 1}] + D_f32[2] * m2_f32[{0, 2}];

                float t_f32 = -Oz_f32 / Dz_f32;
                float u_f32 = Ox_f32 + t_f32 * Dx_f32;
                float v_f32 = Oy_f32 + t_f32 * Dy_f32;

                if (t_f32 > min_distance && t_f32 < intersection.t && t_f32 < max_distance) {

                    if (u_f32 > 0) {

                        if (v_f32 > 0 && u_f32 + v_f32 < 1) {
                            // intersection.state = t < DataT(0) ? Intersection<DataT>::INTERSECTED_BACK :
                            // Intersection<DataT>::INTERSECTED_FRONT;
                            intersection.state = Intersection<DataT>::INTERSECTED_FRONT;
                            intersection.t = t_f32;
                            intersection.barycentric[0] = u_f32;
                            intersection.barycentric[1] = v_f32;
                            intersection.barycentric[2] = 1 - u_f32 - v_f32;
                            intersection.inobject_offset = inobject_offset;
                            intersection.inscene_objid = inscene_objid;

                            if (t > DataT(0.f) && out_t) {
                                *out_t = t_f32;
                            }

                            return true;
                        }
                    }
                }
                return false;
            }

            if (u > DataT(-error_u)) {

                if (v > DataT(-error_v) && u + v < DataT(1.0 + error_u + error_v)) {
                    // intersection.state = t < DataT(0) ? Intersection<DataT>::INTERSECTED_BACK :
                    // Intersection<DataT>::INTERSECTED_FRONT;
                    intersection.state = Intersection<DataT>::INTERSECTED_FRONT;
                    intersection.t = t;
                    intersection.barycentric[0] = u;
                    intersection.barycentric[1] = v;
                    intersection.barycentric[2] = 1 - u - v;
                    intersection.inobject_offset = inobject_offset;
                    intersection.inscene_objid = inscene_objid;

                    if (t > DataT(0.f) && out_t) {
                        *out_t = t;
                    }

                    return true;
                    // return t > DataT(0.f);
                }
            }
        }

        return false;
        
        //auto vao = vaos[vao_index];
        //auto offset = inobject_offset + vao.ebo_offset;

        //auto v0 = vbos[vao.vbo_index][ebos[vao.ebo_index][offset + 0]].position - ray.source;
        //auto v1 = vbos[vao.vbo_index][ebos[vao.ebo_index][offset + 1]].position - ray.source;
        //auto v2 = vbos[vao.vbo_index][ebos[vao.ebo_index][offset + 2]].position - ray.source;

        ///* Calculate triangle edges. */
        //auto e0 = v2 - v0;
        //auto e1 = v0 - v1;
        //auto e2 = v1 - v2;

        ///* Perform edge tests. */
        //auto U = dot_product(cross_product(v2 + v0, e0), ray.direction);
        //auto V = dot_product(cross_product(v0 + v1, e1), ray.direction);
        //auto W = dot_product(cross_product(v1 + v2, e2), ray.direction);

        //auto minUVW = rt::min(U, rt::min(V, W));
        //auto maxUVW = rt::max(U, rt::max(V, W));

        //if constexpr (culling) {
        //    if (minUVW < DataT(0))
        //        return false;
        //} else {
        //    if (minUVW < DataT(0) && maxUVW > DataT(0))
        //        return false;
        //}
        ///* Calculate geometry normal and denominator. */
        //auto Ng1 = cross_product(e1, e0);
        //// const Vec3vfM Ng1 = stable_triangle_normal(e2,e1,e0);
        //auto Ng = Ng1 + Ng1;
        //auto den = dot_product(Ng, ray.direction);

        ///* Avoid division by 0. */
        //if (den == DataT(0.f)) {
        //    return false;
        //}

        ///* Perform depth test. */
        //auto T = dot_product(v0, Ng);
        //auto t = T / den;

        //if (t > min_distance && t < intersection.t && t < max_distance) {
        //    intersection.state =
        //        T < DataT(0) ? Intersection<DataT>::INTERSECTED_BACK : Intersection<DataT>::INTERSECTED_FRONT;
        //    intersection.t = t;
        //    intersection.barycentric[1] = U / den;
        //    intersection.barycentric[2] = V / den;
        //    intersection.barycentric[0] = 1 - intersection.barycentric[1] - intersection.barycentric[2];
        //    intersection.inobject_offset = inobject_offset;
        //    intersection.inscene_objid = inscene_objid;
        //}

        //if (t > DataT(0.f) && out_t) {
        //    *out_t = t;
        //}

        //return t > DataT(0.f);
    }

    template <bool find_any, bool culling>
    __device__ void search_intersection_in_object(const Ray<DataT> &ray, uint32_t inscene_objid,
                                                  uint32_t vao_index, SkipGeometry skip,
                                                  Intersection<DataT> &intersection, DataT min_distance,
                                                  DataT max_distance) const {

        auto bvh = objects_bvh[vao_index];
        uint32_t last_node_id = UINT32_MAX;
        uint32_t current_node_id = 0;

        DataT temp_min_t;
        DataT temp_max_t;

        while (current_node_id < UINT32_MAX) {

            // 访问的节点一定是有效的 ID，但不一定和光线有交点
            auto current_node = bvh.nodes[current_node_id];

            if (last_node_id == current_node.parent) {
                // 如果是从 parent 节点过来的
                // 如果是内部节点

                if (ray_aabb_intersection_object<DataT>(ray, current_node.aabb, &temp_min_t, &temp_max_t) &&
                    temp_min_t < intersection.t && temp_min_t < max_distance && temp_max_t > min_distance) {
                    if (current_node.is_leaf()) {
                        // TODO. 检查一下 AABB，在进去判断相交
                        for (uint32_t i = current_node.geometry_children.offset;
                             i <
                             current_node.geometry_children.offset + current_node.geometry_children.length;
                             ++i) {
                            auto inobject_offset = bvh.geometry_offset[i];
                            if (inobject_offset == skip.inobject_offset && vao_index == skip.vao_index) {
                                continue;
                            }
                            ray_triangle_intersection<culling>(ray, inscene_objid, vao_index, inobject_offset,
                                                               intersection, min_distance, max_distance,
                                                               nullptr);
                            if constexpr (find_any) {
                                if (intersection.intersected()) {
                                    return;
                                }
                            }
                        }
                        last_node_id = current_node_id;
                        current_node_id = current_node.parent;
                    } else {
                        last_node_id = current_node_id;

                        // 如果有交点，则遍历 lc
                        if (current_node.bvh_children.lc < UINT32_MAX) {
                            // 如果有左孩子，则访问左孩子
                            current_node_id = current_node.bvh_children.lc;
                        } else if (current_node.bvh_children.rc < UINT32_MAX) {
                            // 如果没有左孩子，但是有右孩子，就访问右孩子
                            current_node_id = current_node.bvh_children.rc;
                        } else {
                            // 如果没有孩子节点
                            current_node_id = current_node.parent;
                        }
                    }
                } else {
                    // 如果没有交点
                    last_node_id = current_node_id;
                    current_node_id = current_node.parent;
                }

            } else {
                // 从孩子节点回来的
                if (last_node_id == current_node.bvh_children.lc) {

                    // 从 lc 回来的，访问 rc
                    if (current_node.bvh_children.rc < UINT32_MAX) {
                        // 如果有 rc
                        last_node_id = current_node_id;
                        current_node_id = current_node.bvh_children.rc;
                    } else {
                        // 如果没有 rc，直接回到父节点
                        last_node_id = current_node_id;
                        current_node_id = current_node.parent;
                    }

                } else {

                    // 从 rc 回来的，回到父节点
                    last_node_id = current_node_id;
                    current_node_id = current_node.parent;
                }
            }
        }
    }

    // 场景 BVH
    struct {
        typename SceneBVH<DataT>::BVHNode *nodes; // root: 0
        uint32_t *inscene_objid;                  // vao index
        typename SceneBVH<DataT>::AABB *aabbs_W;  // 和 `inscene_objid` 顺序一致的 aabbs，世界坐标

        // visit 的参数是 node id
        // visit 的返回值是检测到的交点位置，TODO 未来可以据此进行剔除
        // 才能进行排除

        // vs find_nearest
        template <bool find_any = false, bool culling = false>
        __device__ void traversal(const Ray<DataT> &ray, const CUDARenderGIInput &input,
                                  Intersection<DataT> &intersection, const SkipGeometry &skip = {},
                                  DataT min_distance = 0, DataT max_distance = DataT(1e5f)) {
            uint32_t last_node_id = UINT32_MAX;
            uint32_t current_node_id = 0;

            DataT temp_min_t = 0;
            DataT temp_max_t = 0;

            while (current_node_id < UINT32_MAX) {

                // 访问的节点一定是有效的 ID，但不一定和光线有交点
                auto current_node = nodes[current_node_id];

                if (last_node_id == current_node.parent) {
                    // 如果是从 parent 节点过来的
                    // 如果是内部节点

                    if (ray_aabb_intersection_scene<DataT>(ray, current_node.aabb, &temp_min_t,
                                                           &temp_max_t) &&
                        temp_min_t < max_distance && temp_max_t > min_distance) {
                        if (current_node.is_leaf()) {
                            for (uint32_t i = current_node.geometry_children.offset;
                                 i < current_node.geometry_children.offset +
                                         current_node.geometry_children.length;
                                 ++i) {
                                if (ray_aabb_intersection_scene<DataT>(ray, aabbs_W[i], &temp_min_t,
                                                                       &temp_max_t) &&
                                    temp_min_t < max_distance && temp_max_t > min_distance) {
                                    auto inscene_objid = this->inscene_objid[i];
                                    auto object_constant = input.objects_constant[inscene_objid];
                                    auto vao_index = object_constant.objectid;
                                    if constexpr (culling) {
                                        if (input.material_constants[object_constant.materialid]
                                                .double_sided) {
                                            input.search_intersection_in_object<find_any, false>(
                                                ray.transformed(object_constant.transform_W2L), inscene_objid,
                                                vao_index, skip, intersection, min_distance, max_distance);
                                        } else {
                                            input.search_intersection_in_object<find_any, true>(
                                                ray.transformed(object_constant.transform_W2L), inscene_objid,
                                                vao_index, skip, intersection, min_distance, max_distance);
                                        }
                                    } else {
                                        input.search_intersection_in_object<find_any, false>(
                                            ray.transformed(object_constant.transform_W2L), inscene_objid,
                                            vao_index, skip, intersection, min_distance, max_distance);
                                    }
                                    if constexpr (find_any) {
                                        if (intersection.intersected()) {
                                            return;
                                        }
                                    }
                                }
                            }
                            last_node_id = current_node_id;
                            current_node_id = current_node.parent;
                        } else {
                            last_node_id = current_node_id;

                            // 如果有交点，则遍历 lc
                            if (current_node.bvh_children.lc < UINT32_MAX) {
                                // 如果有左孩子，则访问左孩子
                                current_node_id = current_node.bvh_children.lc;
                            } else if (current_node.bvh_children.rc < UINT32_MAX) {
                                // 如果没有左孩子，但是有右孩子，就访问右孩子
                                current_node_id = current_node.bvh_children.rc;
                            } else {
                                // 如果没有孩子节点
                                current_node_id = current_node.parent;
                            }
                        }
                    } else {
                        // 如果没有交点
                        last_node_id = current_node_id;
                        current_node_id = current_node.parent;
                    }

                } else {
                    // 从孩子节点回来的
                    if (last_node_id == current_node.bvh_children.lc) {

                        // 从 lc 回来的，访问 rc
                        if (current_node.bvh_children.rc < UINT32_MAX) {
                            // 如果有 rc
                            last_node_id = current_node_id;
                            current_node_id = current_node.bvh_children.rc;
                        } else {
                            // 如果没有 rc，直接回到父节点
                            last_node_id = current_node_id;
                            current_node_id = current_node.parent;
                        }

                    } else {

                        // 从 rc 回来的，回到父节点
                        last_node_id = current_node_id;
                        current_node_id = current_node.parent;
                    }
                }
            }
        }
    } scene_bvh;

    template <bool find_any = false, bool culling = false>
    __device__ Intersection<DataT> cast_ray(const Ray<DataT> &ray, SkipGeometry skip = SkipGeometry{},
                                            DataT min_distance = 0, DataT max_distance = DataT(1e5f)) {
        // 这儿是否需要用 shared memory ?

        Intersection<DataT> intersection;
        // stack less scene bvh traversal
        scene_bvh.traversal<find_any, culling>(ray, *this, intersection, skip, min_distance, max_distance);
        return intersection;
    }
};

// GIData 作为一个资源池，生成 CUDARenderGIInput
template <typename DataT> struct CUDARenderGIData {
    // G-Buffer
    std::shared_ptr<GBuffer> gbuffer;

    // EBO & VBO
    std::shared_ptr<RDResource<DataT>> rd_resource;

    // 若干个 VAO
    thrust::device_vector<CUDAMeshVAO> vaos;

    // Material Constant
    thrust::device_vector<RDLight<DataT>> lights;

    thrust::device_vector<CUDAMaterial<DataT>> materials;

    std::shared_ptr<RDScene<DataT>> rd_scene;

    thrust::device_vector<CUDAObjectConstantData<DataT>> objects_constant;

    Mat4<DataT> transform_W2C;

  private:
    SceneBVH<DataT> scene_bvh;

    thrust::device_vector<typename ObjectBVH<DataT>::Ref> objects_bvh;

  public:
    CUDARenderGIInput<DataT> get_render_input(bool refresh_materials = true) {
        CUDARenderGIInput<DataT> out = {};

        out.gbuffer_color = wrapper_color->get();
        out.gbuffer_inscene_objid = wrapper_objectid->get();
        out.gbuffer_normal_depth = wrapper_normal_depth->get();
        out.gbuffer_position = wrapper_position->get();
        out.gbuffer_tangent = wrapper_tangent->get();
        out.gbuffer_uv0uv1 = wrapper_uv0uv1->get();
        out.gbuffer_inobject_offset = wrapper_inobject_offset->get();

        out.vbos = rd_resource->generate_vbos();
        out.ebos = rd_resource->generate_ebos();
        out.M_shift = rd_resource->generate_m();
        out.M_shift_f32 = rd_resource->generate_mf32();
        out.position_f32 = rd_resource->generate_positionf32();

        if (refresh_materials) {
            this->refresh_materials();
        }
        out.material_constants = thrust::raw_pointer_cast(materials.data());

        out.vaos = thrust::raw_pointer_cast(vaos.data());
        out.n_vaos = vaos.size();

        lights = rd_scene->lights;
        out.lights = thrust::raw_pointer_cast(lights.data());
        out.n_lights = lights.size();

        this->fill_objects_constant();
        out.objects_constant = thrust::raw_pointer_cast(objects_constant.data());

        std::vector<int> vao_indices;
        vao_indices.reserve(rd_scene->objects_constants.size());
        for (auto val : rd_scene->objects_constants) {
            vao_indices.push_back(val.objectid);
        }

        scene_bvh.update(this->rd_scene->objects_aabb, vao_indices);
        out.scene_bvh.inscene_objid = thrust::raw_pointer_cast(this->scene_bvh.inscene_objid.data());
        out.scene_bvh.nodes = thrust::raw_pointer_cast(this->scene_bvh.nodes.data());
        out.scene_bvh.aabbs_W = thrust::raw_pointer_cast(this->scene_bvh.aabbs_W.data());

        out.objects_bvh = thrust::raw_pointer_cast(this->objects_bvh.data());

        out.skybox = rd_scene->skybox.get_ref();

        return out;
    }

    void map() {
        wrapper_color = std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_color()));
        wrapper_objectid = std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_objectid()));
        wrapper_normal_depth =
            std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_normal_depth()));
        wrapper_position = std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_position()));
        wrapper_tangent = std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_tangent()));
        wrapper_uv0uv1 = std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_uv0uv1()));
        wrapper_inobject_offset =
            std::make_unique<CUDASurfaceObjectWrapper>(std::move(gbuffer->get_inobject_offset()));

        for (auto &obj : rd_resource->objects_vbo)
            obj->map();
        for (auto &obj : rd_resource->objects_ebo)
            obj->map();
    }

    void unmap() {
        wrapper_color = nullptr;
        wrapper_objectid = nullptr;
        wrapper_normal_depth = nullptr;
        wrapper_position = nullptr;
        wrapper_tangent = nullptr;
        wrapper_uv0uv1 = nullptr;
        wrapper_inobject_offset = nullptr;

        for (auto &obj : rd_resource->objects_vbo)
            obj->unmap();
        for (auto &obj : rd_resource->objects_ebo)
            obj->unmap();
    }

    void operator=(const CUDARenderGIData<DataT> &data) = delete;
    CUDARenderGIData<DataT>(const CUDARenderGIData<DataT> &) = delete;
    CUDARenderGIData<DataT>() = default;

    // 材质变更（如加载新模型）后重新调用
    void refresh_materials() {
        thrust::host_vector<CUDAMaterial<DataT>> host_constants(rd_resource->materials.size());
        for (int i = 0; i < rd_resource->materials.size(); i++) {
            host_constants[i] = rd_resource->materials[i].constants;
        }
        materials = host_constants;
    }

    // 更新数据后调用一次即可
    void fill_objects_cuda_vao() {
        thrust::host_vector<CUDAMeshVAO> host;
        for (auto vao : rd_resource->objects_vao) {
            host.push_back(CUDAMeshVAO{vao->ebo_index, vao->vbo_index, vao->start_index, vao->n_indices});
        }
        vaos = host;
    }

    // 更新数据后调用一次即可
    void fill_objects_bvh() {
        thrust::host_vector<typename ObjectBVH<DataT>::Ref> host;
        for (auto ptr : rd_resource->objects_bvh) {
            host.push_back(ptr->get_ref());
        }
        this->objects_bvh = host;
    }

  private:
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_color;
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_objectid;
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_normal_depth;
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_position;
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_tangent;
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_uv0uv1;
    std::unique_ptr<CUDASurfaceObjectWrapper> wrapper_inobject_offset;

    // 一般不需要手动调用
    void fill_objects_constant() {
        thrust::host_vector<CUDAObjectConstantData<DataT>> host;
        for (auto val : rd_scene->objects_constants) {
            host.push_back(CUDAObjectConstantData<DataT>(val));
        }
        this->objects_constant = host;
    }
};

template <typename DataT> struct CUDARenderGISettings {
    uint32_t width = 800;
    uint32_t height = 600;
    RDCamera<DataT> camera;

    __device__ Ray<DataT> get_primary_ray(DataT screen_x, DataT screen_y, bool omit_ar = false) {
        DataT normalized_x = DataT(2.0) * (screen_x + DataT(0.5f)) / DataT(width) - DataT(1);
        DataT normalized_y = DataT(2.0) * (screen_y + DataT(0.5f)) / DataT(height) - DataT(1);

        DataT max_y = tan(camera.field_of_view_y / DataT(2.f));
        DataT y = normalized_y * max_y;
        DataT x;
        if (omit_ar) {
            x = normalized_x * max_y * DataT(width) / DataT(height);
        } else {
            x = normalized_x * max_y * camera.aspect_ratio;
        }

        Ray<DataT> ray{Vec3<DataT>(0, 0, 0), Vec3<DataT>(x, y, -1).normalized()};
        ray = ray.transformed(camera.transform_L2W);
        ray.direction.normalize();
        return ray;
    }

    __device__ Vec3<DataT> get_primary_ray_direction(DataT screen_x, DataT screen_y, bool omit_ar = false) {
        DataT normalized_x = DataT(2.0) * (screen_x + DataT(0.5f)) / DataT(width) - DataT(1);
        DataT normalized_y = DataT(2.0) * (screen_y + DataT(0.5f)) / DataT(height) - DataT(1);

        DataT max_y = tan(camera.field_of_view_y / DataT(2.f));
        DataT y = normalized_y * max_y;
        DataT x;
        if (omit_ar) {
            x = normalized_x * max_y * DataT(width) / DataT(height);
        } else {
            x = normalized_x * max_y * camera.aspect_ratio;
        }

        return (camera.transform_L2W * Vec4<DataT>(x, y, -1, 0)).to_vec3_as_dir().normalized();
    }
};

template <typename DataT>
void render_cuda_gi(cudaSurfaceObject_t out, std::shared_ptr<CUDARenderGIData<DataT>> input,
                    CUDARenderGISettings<DataT> settings) {

    int n_total_size = settings.width * settings.height;
    int block_size = 32 * 16; // 不需要共享，设置成一个 warp 的大小就好了
    int grid_size = (n_total_size + block_size - 1) / block_size;

    RT_CHECK_CUDA(
        cudaMemcpyToSymbol(dev_debug_info, &debug_info, sizeof(debug_info), 0, cudaMemcpyHostToDevice))
    input->map();

    test_kernel<<<grid_size, block_size>>>(out, input->get_render_input(), settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    input->unmap();
}

template <typename DataT>
void render_cuda_gbuffer(std::shared_ptr<CUDARenderGIData<DataT>> input, CUDARenderGISettings<DataT> settings,
                         Mat4<DataT> transform_W2C) {

    RT_CHECK_CUDA(cudaGetLastError())

    int n_total_size = settings.width * settings.height;
    int block_size = 32 * 16;
    int grid_size = (n_total_size + block_size - 1) / block_size;

    input->map();
    // TODO 这个地方是个重复调用
    fill_gbuffer_in_cuda<<<grid_size, block_size>>>(input->get_render_input(), settings, transform_W2C);

    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    input->unmap();
}

template <typename DataIn> __device__ DataIn clip_top(DataIn max, DataIn val) {
    if (val < max)
        return val;
    else
        return max;
}

template <typename DataIn> __device__ DataIn clip_both(DataIn min, DataIn max, DataIn val) {
    if (val < min)
        return min;
    else if (val > max)
        return max;
    else
        return val;
}

template <typename DataT> __device__ RGBColor<DataT> to_color_float(uchar4 color) {
    return RGBColor<DataT>(DataT(color.x / 255.f), DataT(color.y / 255.f), DataT(color.z / 255.f));
}

template <typename DataT> __device__ Vec4<DataT> float4_to_vec4(float4 src) {
    return Vec4<DataT>(src.x, src.y, src.z, src.w);
}

template <typename DataT> __device__ uchar4 to_color_uchar(RGBColor<DataT> color) {
    return uchar4{(unsigned char)clip_both<int>(0, 255, int(0.5f + color[0] * DataT(255))),
                  (unsigned char)clip_both<int>(0, 255, int(0.5f + color[1] * DataT(255))),
                  (unsigned char)clip_both<int>(0, 255, int(0.5f + color[2] * DataT(255))), 255};
}

template <typename DataT> __device__ DataT read_surface2d(cudaSurfaceObject_t tex, uint32_t x, uint32_t y) {
    DataT out;
    if constexpr (sizeof(DataT) == 4) {
        *reinterpret_cast<uchar4 *>(&out) =
            surf2Dread<uchar4>(tex, x * sizeof(uchar4), y, cudaBoundaryModeTrap);
    } else if constexpr (sizeof(DataT) == 8) {
        *reinterpret_cast<short4 *>(&out) =
            surf2Dread<short4>(tex, x * sizeof(short4), y, cudaBoundaryModeTrap);
    } else if constexpr (sizeof(DataT) == 16) {
        *reinterpret_cast<float4 *>(&out) =
            surf2Dread<float4>(tex, x * sizeof(float4), y, cudaBoundaryModeTrap);
    } else {
        static_assert(false);
    }
    return out;
}

template <typename DataT>
__device__ void write_surface2d(DataT val, cudaSurfaceObject_t tex, uint32_t x, uint32_t y) {
    if constexpr (sizeof(DataT) == 4) {
        surf2Dwrite(*reinterpret_cast<const uchar4 *>(&val), tex, x * sizeof(uchar4), y);
    } else if constexpr (sizeof(DataT) == 8) {
        surf2Dwrite(*reinterpret_cast<const short4 *>(&val), tex, x * sizeof(short4), y);
    } else if constexpr (sizeof(DataT) == 16) {
        surf2Dwrite(*reinterpret_cast<const float4 *>(&val), tex, x * sizeof(float4), y);
    } else {
        static_assert(false);
    }
}

template <typename DataT> __device__ __host__ constexpr DataT RAY_MOVEFORWARD_T() {
    static_assert(sizeof(DataT) == 2 || sizeof(DataT) == 4);

    if constexpr (sizeof(DataT) == 4) {
        return 1e-4f;
    } else {
        return DataT(1e-1);
    }
}

template <int N, typename T>
__device__ Vector<N, T> lerp(const Vec3<T> &barycentric, const Vector<N, T> &v1, const Vector<N, T> &v2,
                             const Vector<N, T> &v3) {
    return barycentric[0] * v1 + barycentric[1] * v2 + barycentric[2] * v3;
}

template <typename DataT>
__global__ void fill_gbuffer_in_cuda(CUDARenderGIInput<DataT> input, CUDARenderGISettings<DataT> settings,
                                     Mat4<DataT> world_to_clip) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;
    Ray<DataT> primary_ray = settings.get_primary_ray(x, y, true);
    auto view_dir = -primary_ray.direction;
    Intersection<DataT> intersection(input.cast_ray<false, true>(primary_ray));

    // 会做统一 clear

    if (intersection.intersected()) {

        const CUDAMeshVAO vao = input.vaos[input.objects_constant[intersection.inscene_objid].objectid];
        const RDVertex<DataT> *vbo = input.vbos[vao.vbo_index];
        const uint32_t *ebo = input.ebos[vao.ebo_index];

        RDVertex<DataT> v1 = vbo[ebo[vao.ebo_offset + intersection.inobject_offset + 0]];
        RDVertex<DataT> v2 = vbo[ebo[vao.ebo_offset + intersection.inobject_offset + 1]];
        RDVertex<DataT> v3 = vbo[ebo[vao.ebo_offset + intersection.inobject_offset + 2]];

        auto position = lerp(intersection.barycentric, v1.position, v2.position, v3.position);
        //    auto position = primary_ray.source + intersection.t * primary_ray.direction;
        auto normal = lerp(intersection.barycentric, v1.normal, v2.normal, v3.normal).normalized();
        auto color = lerp(intersection.barycentric, v1.color, v2.color, v3.color);
        auto tangent = lerp(intersection.barycentric, v1.tangent, v2.tangent, v3.tangent).normalized();
        auto uv0 = lerp(intersection.barycentric, v1.uv0, v2.uv0, v3.uv0);
        auto uv1 = lerp(intersection.barycentric, v1.uv1, v2.uv1, v3.uv1);

        auto transform = input.objects_constant[intersection.inscene_objid].transform_L2W;
        normal = (transform * normal.to_vec4_as_dir()).to_vec3_as_dir().normalized();
        tangent = (transform * tangent.to_vec4_as_dir()).to_vec3_as_dir().normalized();
        auto position4 = (transform * position.to_vec4_as_pos());
        position = position4.to_vec3_as_pos();
        /* printf("trans: %.3f, %.3f, %.3f, position: " RT_VEC3_F "\n", transform[{0, 3}], transform[{1, 3}],
                transform[{2, 3}], RT_VEC3(position));*/

        write_surface2d<Vec4<DataT>>(Vec4<DataT>(position[0], position[1], position[2], 1),
                                     input.gbuffer_position, x, y);
        write_surface2d<Vec4<DataT>>(Vec4<DataT>(normal[0], normal[1], normal[2], DataT(1.f)),
                                     input.gbuffer_normal_depth, x, y);
        write_surface2d<Vec4<DataT>>(Vec4<DataT>(tangent[0], tangent[1], tangent[2], DataT(1.f)),
                                     input.gbuffer_tangent, x, y);
        write_surface2d<uchar4>(to_color_uchar(color), input.gbuffer_color, x, y);
        write_surface2d(Vec4<DataT>(uv0[0], uv0[1], uv1[0], uv1[1]), input.gbuffer_uv0uv1, x, y);

        write_surface2d<uint32_t>(intersection.inscene_objid, input.gbuffer_inscene_objid, x, y);
        write_surface2d<uint32_t>(intersection.inobject_offset, input.gbuffer_inobject_offset, x, y);
    } else {
        write_surface2d<Vec4<DataT>>(Vec4<DataT>(0, 0, 0, 0), input.gbuffer_position, x, y);
        write_surface2d<Vec4<DataT>>(Vec4<DataT>(0, 0, 0, 0), input.gbuffer_normal_depth, x, y);
        write_surface2d<Vec4<DataT>>(Vec4<DataT>(0, 0, 0, 0), input.gbuffer_tangent, x, y);
        write_surface2d<uchar4>(uchar4{0, 0, 0, 0}, input.gbuffer_color, x, y);
        write_surface2d(Vec4<DataT>(0, 0, 0, 0), input.gbuffer_uv0uv1, x, y);

        write_surface2d<uint32_t>(0, input.gbuffer_inscene_objid, x, y);
        write_surface2d<uint32_t>(0, input.gbuffer_inobject_offset, x, y);
    }
}

// 追踪，用于 GI（需要找到最近的交点）
template <typename DataT> struct TraceGICommand {
    uint16_t x;
    uint16_t y;
    Vec3<DataT> source;
    Vec3<DataT> direction;
};

// 追踪，用于直接光照（只要知道有没有交点即可）
template <typename DataT> struct TraceLightCommand {
    uint16_t x;
    uint16_t y;
    Vec3<DataT> source;
    Vec3<DataT> direction;
    DataT maximum_t;
    uint8_t slot; // slot == uint8_max: 无效，不需要做。注意，需要清空 di_intensity !
    Vec3<DataT> material_multiplier;
    typename CUDARenderGIInput<DataT>::SkipGeometry skip;
};

template <typename DataT> struct PixelShaderInput {
    enum Type { INVALID, COMMON, SKYBOX } type = INVALID;

    Vec3<DataT> color;
    Vec3<DataT> normal;
    Vec3<DataT> position;
    Vec3<DataT> tangent;
    Vec2<DataT> uv0;
    Vec2<DataT> uv1;

    uint32_t materialid;

    uint32_t inscene_objid;
    uint32_t inobject_offset;
};

struct LastFramePixelID {
    uint32_t object_id;
    uint32_t inobject_offset;
};

template <typename DataT> struct TemporalMap {
    uint8_t frame_count;

    // 屏幕空间坐标
    // 根据 SVGF 论文的描述，
    // 如果在 2x2 范围内可以找到和上一帧一致的，则在 2x2 范围内找
    // 否则，将在 3x3 范围内找
    // 所以，一般最多采样 4 个坐标
    Vec2<uint16_t> last_frame_pos[4];
    DataT weights[4] = {};

    // template <typename T>
    //__device__ T sample_nearest(const T *arr, int width, int height, const T &default_value) {
    //    // y *width + x;
    //    if (frame_count == 0) {
    //        return default_value;
    //    }

    //    auto last_frame_x = this->last_frame_x - DataT(0.5);
    //    auto last_frame_y = this->last_frame_y - DataT(0.5);

    //    auto round_last_frame_x = clip_both<uint32_t>(lround(last_frame_x), 0, width - 1);
    //    auto round_last_frame_y = clip_both<uint32_t>(lround(last_frame_x), 0, height - 1);

    //    uint32_t bb = round_last_frame_x + round_last_frame_y * width;

    //    return arr[bb];
    //}
};

template <typename DataT, typename T, typename ScalarType, typename ArrayType,
          bool IS_SURFACE =
              std::is_same_v<cudaSurfaceObject_t, std::remove_reference_t<std::remove_cv_t<ArrayType>>>,
          bool IS_ARRAY = std::is_same_v<T *, std::remove_reference_t<std::remove_cv_t<ArrayType>>>,
          typename VALID_T = std::enable_if_t<IS_SURFACE || IS_ARRAY>>
__device__ T sample_linear(const TemporalMap<DataT> &map, ArrayType arr, int width, int height,
                           const T &default_value) {
    // y *width + x;
    if (map.frame_count == 0) {
        return default_value;
    }

    DataT sum_weight = 0;
    T result = {};
    for (int i = 0; i < 4; i++) {
        auto pos = map.last_frame_pos[i];
        auto weight = map.weights[i];
        if (weight == DataT(0)) {
            continue;
        }
        sum_weight += weight;
        if constexpr (IS_ARRAY) {
            auto q_tid = pos[1] * width + pos[0];
            result += weight * arr[q_tid];
        } else {
            result += weight * read_surface2d<T>(arr, pos[0], pos[1]);
        }
    }

    return result / sum_weight;
}

template <typename DataT> struct SVGFBuffer {
    thrust::device_vector<DataT> miu1_1;
    thrust::device_vector<DataT> miu2_1;
    thrust::device_vector<DataT> miu1_2;
    thrust::device_vector<DataT> miu2_2;
    thrust::device_vector<DataT> var_1;
    thrust::device_vector<DataT> var_2;
    thrust::device_vector<Vec3<DataT>> color_history;
    thrust::device_vector<Vec3<DataT>> color_buffer;

    thrust::device_vector<Vec3<DataT>> normal_buffer;
    thrust::device_vector<DataT> depth_buffer;
    thrust::device_vector<Vec2<DataT>> depth_gradient;
    thrust::device_vector<DataT> illum_buffer;

    struct Ref {
        // TODO 这里的 miux_2 和 var_2 所有 SVGFBuffer 共享一份就够用了
        DataT *miu1_1;
        DataT *miu2_1;
        DataT *miu1_2;
        DataT *miu2_2;
        DataT *var_1;
        DataT *var_2;
        Vec3<DataT> *color_history;
        Vec3<DataT> *color_buffer;

        Vec3<DataT> *normal_buffer;
        DataT *depth_buffer;
        Vec2<DataT> *depth_gradient;
        DataT *illum_buffer;
    };

    SVGFBuffer(int width, int height)
        : miu1_1(width * height), miu2_1(width * height), miu1_2(width * height), miu2_2(width * height),
          var_1(width * height), var_2(width * height), color_history(width * height),
          color_buffer(width * height), normal_buffer(width * height), depth_buffer(width * height),
          depth_gradient(width * height), illum_buffer(width * height)

    {}

    Ref get_ref() {
        return {
            thrust::raw_pointer_cast(miu1_1.data()),         thrust::raw_pointer_cast(miu2_1.data()),
            thrust::raw_pointer_cast(miu1_2.data()),         thrust::raw_pointer_cast(miu2_2.data()),
            thrust::raw_pointer_cast(var_1.data()),          thrust::raw_pointer_cast(var_2.data()),
            thrust::raw_pointer_cast(color_history.data()),  thrust::raw_pointer_cast(color_buffer.data()),
            thrust::raw_pointer_cast(normal_buffer.data()),  thrust::raw_pointer_cast(depth_buffer.data()),
            thrust::raw_pointer_cast(depth_gradient.data()), thrust::raw_pointer_cast(illum_buffer.data())};
    }
};

template <int M, int N, typename DataT>
void __device__ value_or(Matrix<M, N, DataT> &inout, const Matrix<M, N, DataT> &default_value) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (!isfinite(inout.data[i][j])) {
                inout.data[i][j] = default_value.data[i][j];
            }
        }
    }
}

// Figure 3 中的 Temporal Accumulation 过程
// * color 计算好，存到 `integrated_color` 中
// * miu1 & miu2 计算好，存到 `miu1_out`, `miu2_out` 中
// * integrated moments 计算好，存到 `var_out` 中
// * illum 计算好，存到 `illum_out` 中
// input color: 这一帧的信息
// color_inprogress color：以前的信息
// integrated color：参照 Figure 3. 中间值。可以和 input_color 指定为同一个数组
template <typename DataT>
__global__ void temporal_accumulation_color(Vec3<DataT> *input_color, Vec3<DataT> *history_color,
                                            TemporalMap<DataT> *svgf_temporal_map, Vec3<DataT> *integrated_color,
                                            DataT *illum_out, CUDARenderGISettings<DataT> settings,
                                            DataT color_taa_w) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    int p_x = thread_id % settings.width;
    int p_y = thread_id / settings.width;

    TemporalMap<DataT> temporal_info = svgf_temporal_map[thread_id];

    // 0. Remove Outlier (refer to GAMES 202)
    Vec3<DataT> p_color = input_color[thread_id];
    Vec3<DataT> p_miu1 = {}, p_miu2 = {};
    Vec3<DataT> weights = {};
    for (int i = -4; i <= 4; i++) {
        for (int j = -4; j <= 4; j++) {
            int q_x = p_x + i;
            int q_y = p_y + j;
            if (q_x < 0 || q_x >= settings.width || q_y < 0 || q_y >= settings.height) {
                continue;
            }
            int q_tid = q_y * settings.width + q_x;

            Vec3<DataT> q_color = input_color[q_tid];
            for (int k = 0; k < 3; k++) {
                if (isfinite(q_color[k])) {
                    weights[k] += 1;
                    p_miu1[k] += q_color[k];
                    p_miu2[k] += q_color[k] * q_color[k];
                }
            }
        }
    }

    p_miu1 = elediv(p_miu1, weights);
    p_miu2 = elediv(p_miu2, weights);
    value_or(p_color, p_miu1);
    Vec3<DataT> var = p_miu2 - elemul(p_miu1, p_miu1);
    for (int i = 0; i < 3; i++) {
        DataT std = sqrt(var[i]);
        p_color[i] = clip_both(p_miu1[i] - DataT(0.5) * std, p_miu1[i] + DataT(0.5) * std, p_color[i]);
    }

    // 1. Integrated Color
    Vec3<DataT> color_history = sample_linear<DataT, Vec3<DataT>, DataT, Vec3<DataT> *>(
        temporal_info, history_color, settings.width, settings.height, p_color);
    value_or(color_history, p_color);
    integrated_color[thread_id] = color_taa_w * p_color + (DataT(1) - color_taa_w) * color_history;
    DataT illum = DataT(0.2126) * integrated_color[thread_id][0] +
                  DataT(0.7152) * integrated_color[thread_id][1] +
                  DataT(0.0722) * integrated_color[thread_id][2];
    illum_out[thread_id] = illum;
}

template <typename DataT>
__global__ void temporal_accumulation_moments(TemporalMap<DataT> *svgf_temporal_map, DataT *depth,
                                              Vec2<DataT> *depth_gradient, Vec3<DataT> *world_normal,
                                              DataT *miu1, DataT *miu2, DataT *miu1_out, DataT *miu2_out,
                                              DataT *var_out, DataT *illum_in,
                                              CUDARenderGISettings<DataT> settings, DataT moments_taa_w) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    int p_x = thread_id % settings.width;
    int p_y = thread_id / settings.width;

    const DataT SIGMA_Z = 1;
    const DataT SIGMA_N = 128;
    const DataT SIGMA_L = 4;
    const DataT EPS = 1e-5; // TODO

    TemporalMap<DataT> temporal_info = svgf_temporal_map[thread_id];

    DataT illum = illum_in[thread_id];
    DataT illum2 = illum * illum;
    DataT wavelet_h[3] = {DataT(3. / 8), DataT(1. / 4), DataT(1. / 16)};

    // 2. Integrated Moments
    DataT miu1_val = 0, miu2_val = 0;
    if (temporal_info.frame_count < 4) {
        // 7x7 bilateral filter with weights driven by depths and world space normals
        // TODO 7x7 ?

        DataT depth_p = depth[thread_id];
        Vec2<DataT> depth_gradient_value = depth_gradient[thread_id];
        Vec3<DataT> normal_p = world_normal[thread_id];
        DataT weight = 0;

        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int q_x = p_x + i;
                int q_y = p_y + j;
                if (q_x < 0 || q_x >= settings.width || q_y < 0 || q_y >= settings.height) {
                    continue;
                }

                int q_tid = q_y * settings.width + q_x;

                auto h_value = wavelet_h[abs(i)] * wavelet_h[abs(j)];

                DataT depth_q = depth[q_tid];
                Vec2<DataT> dp = Vec2<DataT>(DataT(i), DataT(j));
                DataT w_z = exp(-abs(depth_p - depth_q) /
                                (SIGMA_Z * abs(dot_product(depth_gradient_value, dp) + EPS)));

                Vec3<DataT> normal_q = world_normal[q_tid];
                DataT w_n = pow(max(DataT(0), dot_product(normal_p, normal_q)), SIGMA_N);

                DataT w_value = w_z * w_n;
                DataT hw_value = h_value * w_value;

                if (isfinite(hw_value)) {
                    DataT illum_q = illum_in[q_tid];
                    if (isfinite(illum_q)) {
                        miu1_val += hw_value * illum_q;
                        miu2_val += hw_value * illum_q * illum_q;
                        weight += hw_value;
                    }
                }
            }
        }

        miu1_val /= weight;
        miu2_val /= weight;

    } else {
        miu1_val = (DataT(1) - moments_taa_w) *
                       sample_linear<DataT, DataT, DataT, DataT *>(temporal_info, miu1, settings.width,
                                                                   settings.height, miu1_val) +
                   illum * moments_taa_w;
        miu2_val = (DataT(1) - moments_taa_w) *
                       sample_linear<DataT, DataT, DataT, DataT *>(temporal_info, miu2, settings.width,
                                                                   settings.height, miu2_val) +
                   illum2 * moments_taa_w;
        if (!isfinite(miu1_val)) {
            miu1_val = illum;
        }
        if (!isfinite(miu2_val)) {
            miu2_val = illum2;
        }
    }
    DataT variance = miu2_val - miu1_val * miu1_val;
    miu1_out[thread_id] = miu1_val;
    miu2_out[thread_id] = miu2_val;
    var_out[thread_id] = variance;
}

template <typename DataT, int stride>
__global__ void wavelet_filter(Vec3<DataT> *color_in, Vec3<DataT> *color_out, DataT *var_in, DataT *var_out,
                               DataT *depth, Vec2<DataT> *depth_gradient, Vec3<DataT> *world_normal,
                               DataT *illum, CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    int p_x = thread_id % settings.width;
    int p_y = thread_id / settings.width;

    const DataT SIGMA_Z = 1;
    const DataT SIGMA_N = 128;
    const DataT SIGMA_L = 4;
    const DataT EPS = 1e-5; // TODO

    DataT wavelet_h[3] = {DataT(3. / 8), DataT(1. / 4), DataT(1. / 16)};
    DataT gaussian_g[2] = {DataT(1. / 2), DataT(1. / 4)};

    // 1. Gaussian on variance
    DataT sqrt_g3x3_var_p = 0;
    DataT g3x3_var_weight = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int q_x = p_x + i;
            int q_y = p_y + j;
            if (q_x < 0 || q_x >= settings.width || q_y < 0 || q_y >= settings.height) {
                continue;
            }
            int q_tid = q_y * settings.width + q_x;
            DataT g_value = gaussian_g[abs(i)] * gaussian_g[abs(j)];
            sqrt_g3x3_var_p += g_value * var_in[q_tid];
            g3x3_var_weight += g_value;
        }
    }
    sqrt_g3x3_var_p = sqrt(sqrt_g3x3_var_p / g3x3_var_weight);

    // 2. filtered color
    Vec3<DataT> normal_p = world_normal[thread_id];
    DataT next_var_p = 0;
    Vec3<DataT> next_color_p = {};
    DataT weight_sum_var_p = 0;
    DataT weight_sum_color_p = 0;

    DataT depth_p = depth[thread_id];
    DataT illum_p = illum[thread_id];
    Vec2<DataT> depth_gradient_value = depth_gradient[thread_id];
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int q_x = p_x + stride * i;
            int q_y = p_y + stride * j;
            if (q_x < 0 || q_x >= settings.width || q_y < 0 || q_y >= settings.height) {
                continue;
            }

            int q_tid = q_y * settings.width + q_x;

            auto h_value = wavelet_h[abs(i)] * wavelet_h[abs(j)];

            DataT depth_q = depth[q_tid];
            Vec2<DataT> dp = Vec2<DataT>(DataT(stride * i), DataT(stride * j));
            DataT w_z =
                exp(-abs(depth_p - depth_q) / (SIGMA_Z * abs(dot_product(depth_gradient_value, dp) + EPS)));

            Vec3<DataT> normal_q = world_normal[q_tid];
            DataT w_n = pow(max(DataT(0), dot_product(normal_p, normal_q)), SIGMA_N);

            DataT illum_q = illum[q_tid];
            DataT w_l = exp(-abs(illum_p - illum_q) / (SIGMA_L * sqrt_g3x3_var_p + EPS));

            DataT w_value = w_z * w_n * w_l;
            DataT hw_value = h_value * w_value;

            Vec3<DataT> color_q = color_in[q_tid];

            DataT var_q = var_in[q_tid];

            if (isfinite(hw_value)) {
                if (isfinite(var_q)) {
                    next_var_p += hw_value * hw_value * var_q;
                    weight_sum_var_p += hw_value;
                }

                if (isfinite(color_q[0]) && isfinite(color_q[1]) && isfinite(color_q[2])) {
                    next_color_p += hw_value * color_q;
                    weight_sum_color_p += hw_value;
                }
            }
        }
    }

    next_var_p /= (weight_sum_var_p * weight_sum_var_p);
    next_color_p /= weight_sum_color_p;

    value_or(next_color_p, color_in[thread_id]);
    if (!isfinite(next_var_p)) {
        next_var_p = var_in[thread_id];
    }

    color_out[thread_id] = next_color_p;
    var_out[thread_id] = next_var_p;
}

template <typename DataT>
__global__ void preprocess_normal_depth(cudaSurfaceObject_t normal_depth, Vec3<DataT> *out_normal,
                                        DataT *out_depth, Vec2<DataT> *out_depth_gradient,
                                        CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    const int p_x = thread_id % settings.width;
    const int p_y = thread_id / settings.width;

    Vec4<DataT> normal_depth_val = read_surface2d<Vec4<DataT>>(normal_depth, p_x, p_y);
    out_normal[thread_id] = normal_depth_val.xyz();

    DataT depth_p = normal_depth_val[3];
    out_depth[thread_id] = depth_p;

    Vec2<DataT> gradient = {};
    if (p_x == 0) {
        gradient[0] = read_surface2d<Vec4<DataT>>(normal_depth, p_x + 1, p_y)[3] - depth_p;
    } else {
        gradient[0] = depth_p - read_surface2d<Vec4<DataT>>(normal_depth, p_x - 1, p_y)[3];
    }

    if (p_y == 0) {
        gradient[1] = read_surface2d<Vec4<DataT>>(normal_depth, p_x, p_y + 1)[3] - depth_p;
    } else {
        gradient[1] = depth_p - read_surface2d<Vec4<DataT>>(normal_depth, p_x, p_y - 1)[3];
    }

    out_depth_gradient[thread_id] = gradient;
}

template <typename DataT, size_t MAX_DIRECT_LIGHT> struct RTRTProcedureBuffer {

    // SIZE，记录有几个 previous frame
    thrust::device_vector<TemporalMap<DataT>> svgf_temporal_map;
    thrust::device_vector<TemporalMap<DataT>> taa_temporal_map;
    thrust::device_vector<uint8_t> valid_frame_count;
    thrust::device_vector<Mat4<DataT>> last_frame_L2W;
    thrust::device_vector<LastFramePixelID> last_frame_id;

    // (N-1) x SIZE。
    // intensity 表示迄今为止的光强。直接光计算完成后直接填充进去，间接光递归完成后再填充
    // gi_material_multiplier 用于在递归回来后计算光线强度
    thrust::device_vector<Vec3<DataT>> gi_material_multiplier;

    // 存储 di_intensity 之和
    // N x SIZE。
    thrust::device_vector<Vec3<DataT>> intensity;

    // SIZE
    // 本次 shade 的 view direction，由上一轮 shade 填充
    // 直接覆盖即可
    thrust::device_vector<Vec3<DataT>> view_direction;
    // SIZE
    thrust::device_vector<typename CUDARenderGIInput<DataT>::SkipGeometry> skips;

    // 根据 barycentric 计算相应信息可以在 shade 或者 trace 中去做
    // 暂时决定在 trace 中做，shade 保持干净。做成 AoS
    // 需要清0
    // SIZE
    thrust::device_vector<PixelShaderInput<DataT>> shade_commands;

    // SIZE * MAX_DIRECT_LIGHT 直接光光追命令
    thrust::device_vector<TraceLightCommand<DataT>> trace_light_commands;

    // SIZE X MAX_DIRECT_LIGHT
    // 直接光结果，需要清零
    thrust::device_vector<Vec3<DataT>> di_intensity;

    // SIZE 间接光光追命令
    thrust::device_vector<TraceGICommand<DataT>> trace_gi_commands;

    // SIZE
    thrust::device_vector<RGBColor<DataT>> albedo;
    thrust::device_vector<RGBColor<DataT>> mul_gi_colored;
    thrust::device_vector<RGBColor<DataT>> mul_gi_white;

    thrust::device_vector<RGBColor<DataT>> color_in_progress;

    SVGFBuffer<DataT> gi_colored_svgf;
    SVGFBuffer<DataT> gi_white_svgf;

    thrust::device_vector<RGBColor<DataT>> taa_history_color;

    int object_count;
    int width;
    int height;
    int max_bounces;

  public:
    Mat4<DataT> last_frame_transform_W2C;

  private:
  public:
    struct Ref {
        Vec3<DataT> *gi_material_multiplier;
        Vec3<DataT> *intensity;
        Vec3<DataT> *view_direction;
        typename CUDARenderGIInput<DataT>::SkipGeometry *skips;
        PixelShaderInput<DataT> *shade_commands;
        TraceLightCommand<DataT> *trace_light_commands;
        Vec3<DataT> *di_intensity;
        TraceGICommand<DataT> *trace_gi_command;
        RGBColor<DataT> *albedo;
        RGBColor<DataT> *mul_gi_colored;
        RGBColor<DataT> *mul_gi_white;
        TemporalMap<DataT> *svgf_temporal_map;
        TemporalMap<DataT> *taa_temporal_map;
        Mat4<DataT> *last_frame_L2W;
        LastFramePixelID *last_frame_id;
        uint8_t *valid_frame_count;
        RGBColor<DataT> *color_inprogress;
        typename SVGFBuffer<DataT>::Ref gi_colored_svgf;
        typename SVGFBuffer<DataT>::Ref gi_white_svgf;
        RGBColor<DataT> *taa_history_color;
        int slice_size;
        Mat4<DataT> last_frame_transform_W2C;
    };

    void clear_intensity() {
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(intensity.data()), 0,
                                 intensity.size() * sizeof(Vec3<DataT>)));
    }

    void clear_trace_gi_commands() {
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(trace_gi_commands.data()), 0,
                                 trace_gi_commands.size() * sizeof(TraceGICommand<DataT>)));
    }

    void clear_di_intensity() {
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(di_intensity.data()), 0,
                                 di_intensity.size() * sizeof(Vec3<DataT>)));
    }

    void clear_shade_commands() {
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(shade_commands.data()), 0,
                                 shade_commands.size() * sizeof(PixelShaderInput<DataT>)));
    }

    void clear_filterable_cache() {
        RT_CHECK_CUDA(
            cudaMemset(thrust::raw_pointer_cast(albedo.data()), 0, albedo.size() * sizeof(RGBColor<DataT>)));
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(mul_gi_colored.data()), 0,
                                 mul_gi_colored.size() * sizeof(RGBColor<DataT>)));
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(mul_gi_white.data()), 0,
                                 mul_gi_white.size() * sizeof(RGBColor<DataT>)));
    }

    void clear_last_frame_L2W() {
        RT_CHECK_CUDA(cudaMemset(thrust::raw_pointer_cast(last_frame_L2W.data()), 0,
                                 last_frame_L2W.size() * sizeof(Mat4<DataT>)));
    }

    RTRTProcedureBuffer(const RTRTProcedureBuffer &) = delete;
    void operator=(const RTRTProcedureBuffer &) = delete;
    RTRTProcedureBuffer(int width, int height, int max_bounces, int object_count)
        : width(width), height(height), max_bounces(max_bounces), object_count(object_count),
          gi_material_multiplier((max_bounces - 1) * width * height), intensity(max_bounces * width * height),
          view_direction(width * height), skips(width * height), shade_commands(width * height),
          trace_light_commands(width * height * MAX_DIRECT_LIGHT),
          di_intensity(width * height * MAX_DIRECT_LIGHT), trace_gi_commands(width * height),
          albedo(width * height), mul_gi_colored(width * height), mul_gi_white(width * height),
          svgf_temporal_map(width * height), taa_temporal_map(width * height), last_frame_L2W(object_count), last_frame_id(width * height),
          color_in_progress(width * height), gi_colored_svgf(width, height), gi_white_svgf(width, height),
          valid_frame_count(width * height), taa_history_color(width * height) {}

    Ref get_ref() {
        return {thrust::raw_pointer_cast(gi_material_multiplier.data()),
                thrust::raw_pointer_cast(intensity.data()),
                thrust::raw_pointer_cast(view_direction.data()),
                thrust::raw_pointer_cast(skips.data()),
                thrust::raw_pointer_cast(shade_commands.data()),
                thrust::raw_pointer_cast(trace_light_commands.data()),
                thrust::raw_pointer_cast(di_intensity.data()),
                thrust::raw_pointer_cast(trace_gi_commands.data()),
                thrust::raw_pointer_cast(albedo.data()),
                thrust::raw_pointer_cast(mul_gi_colored.data()),
                thrust::raw_pointer_cast(mul_gi_white.data()),
                thrust::raw_pointer_cast(svgf_temporal_map.data()),
                thrust::raw_pointer_cast(taa_temporal_map.data()),
                thrust::raw_pointer_cast(last_frame_L2W.data()),
                thrust::raw_pointer_cast(last_frame_id.data()),
                thrust::raw_pointer_cast(valid_frame_count.data()),
                thrust::raw_pointer_cast(color_in_progress.data()),
                gi_colored_svgf.get_ref(),
                gi_white_svgf.get_ref(),
                thrust::raw_pointer_cast(taa_history_color.data()),
                width * height,
                last_frame_transform_W2C};
    }
};

template <typename DataT, int MAX_DIRECT_LIGHT>
void svgf_denoise(Vec3<DataT> *color_inout, typename SVGFBuffer<DataT>::Ref svgf,
                  CUDARenderGIInput<DataT> input,
                  typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                  CUDARenderGISettings<DataT> settings, DataT taa_color_w, DataT taa_moments_w) {

    int n_total_size = settings.width * settings.height;
    int block_size = 32 * 2;
    int grid_size = (n_total_size + block_size - 1) / block_size;

    preprocess_normal_depth<<<grid_size, block_size>>>(input.gbuffer_normal_depth, svgf.normal_buffer,
                                                       svgf.depth_buffer, svgf.depth_gradient, settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    // temporal accumulation & variance estimation
    temporal_accumulation_color<DataT><<<grid_size, block_size>>>(color_inout, svgf.color_history,
                                                                  buffer.svgf_temporal_map, color_inout,
                                                                  svgf.illum_buffer, settings, taa_color_w);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    temporal_accumulation_moments<DataT><<<grid_size, block_size>>>(
        buffer.svgf_temporal_map, svgf.depth_buffer, svgf.depth_gradient, svgf.normal_buffer, svgf.miu1_1,
        svgf.miu2_1, svgf.miu1_2, svgf.miu2_2, svgf.var_1, svgf.illum_buffer, settings, taa_moments_w);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    // update moments color_inprogress
    // TODO 这段内存拷贝可以去除, swap 一下就好啦
    RT_CHECK_CUDA(
        cudaMemcpy(svgf.miu1_1, svgf.miu1_2, settings.width * settings.height, cudaMemcpyDeviceToDevice))
    RT_CHECK_CUDA(
        cudaMemcpy(svgf.miu2_1, svgf.miu2_2, settings.width * settings.height, cudaMemcpyDeviceToDevice))

    // Current State
    // Integrated Color: color_inout
    // Integrated Moments: svgf.miu1_2, svgf.miu2_2
    // Variance: svgf.var_1

    // wavelet filter iteration #1
    // Update color color_inprogress
    wavelet_filter<DataT, 1><<<grid_size, block_size>>>(color_inout, svgf.color_history, svgf.var_1,
                                                        svgf.var_2, svgf.depth_buffer, svgf.depth_gradient,
                                                        svgf.normal_buffer, svgf.illum_buffer, settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    // wavelet filter iteration #2
    wavelet_filter<DataT, 2><<<grid_size, block_size>>>(svgf.color_history, svgf.color_buffer, svgf.var_2,
                                                        svgf.var_1, svgf.depth_buffer, svgf.depth_gradient,
                                                        svgf.normal_buffer, svgf.illum_buffer, settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    // wavelet filter iteration #3
    wavelet_filter<DataT, 4><<<grid_size, block_size>>>(svgf.color_buffer, color_inout, svgf.var_1,
                                                        svgf.var_2, svgf.depth_buffer, svgf.depth_gradient,
                                                        svgf.normal_buffer, svgf.illum_buffer, settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    // wavelet filter iteration #4
    wavelet_filter<DataT, 8><<<grid_size, block_size>>>(color_inout, svgf.color_buffer, svgf.var_2,
                                                        svgf.var_1, svgf.depth_buffer, svgf.depth_gradient,
                                                        svgf.normal_buffer, svgf.illum_buffer, settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())

    // wavelet filter iteration #5
    wavelet_filter<DataT, 16><<<grid_size, block_size>>>(svgf.color_buffer, color_inout, svgf.var_1,
                                                         svgf.var_2, svgf.depth_buffer, svgf.depth_gradient,
                                                         svgf.normal_buffer, svgf.illum_buffer, settings);
    RT_CHECK_CUDA(cudaDeviceSynchronize())
    RT_CHECK_CUDA(cudaGetLastError())
}

// reference: blender
template <typename DataT> __device__ Vec3<DataT> sample_ggx(DataT a2, curandState_t &state) {
    DataT rand_x = curand_uniform(&state);
    DataT rand_y = curand_uniform(&state);
    DataT rand_z = curand_uniform(&state);
    DataT z = sqrt((DataT(1) - rand_x) / (DataT(1) + a2 * rand_x - rand_x)); /* cos theta */
    DataT r = sqrt(max(DataT(0), DataT(1) - z * z));                         /* sin theta */
    DataT x = r * rand_y;
    DataT y = r * rand_z;
    return Vec3<DataT>(x, y, z);
}

// reference: blender
template <typename DataT> __device__ DataT D_ggx_opti(DataT NH, DataT a2) {
    DataT tmp = (NH * a2 - NH) * NH + DataT(1.0);
    return M_PI * tmp * tmp;
}

// reference: blender
template <typename DataT> __device__ DataT pdf_ggx_reflect(DataT NH, DataT a2) {
    return NH * a2 / D_ggx_opti(NH, a2);
}

template <typename DataT>
__device__ Vec3<DataT> tangent_to_world(const Vec3<DataT> &vec, const Vec3<DataT> &N, const Vec3<DataT> &T,
                                        const Vec3<DataT> &B) {
    return T * vec[0] + B * vec[1] + N * vec[2];
}

template <typename DataT>
__device__ Vec2<float> direction_to_spherical(const Vec3<DataT> &dir, DataT offset_x, DataT offset_y) {
    Vec2<float> uv(float(0.1591f * atan2(float(dir[1]), float(dir[0])) + 0.5f + float(offset_x)),
                   float(0.3183f * asin(float(dir[2])) + 0.5f + float(offset_y)));
    uv[0] = fmodf(uv[0], 1);
    uv[1] = 1 - fmodf(uv[1], 1);
    return uv;
}

template <typename DataT, size_t MAX_DIRECT_LIGHT, bool FIRST_ROUND, bool NO_GI>
__global__ void shade(CUDARenderGIInput<DataT> input,
                      typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer, int round,
                      uint64_t seed, CUDARenderGISettings<DataT> settings) {

    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    curandState_t rand_state;
    curand_init(seed, thread_id, 0, &rand_state);

    Vec3<DataT> position;
    Vec3<DataT> raw_normal;
    Vec3<DataT> raw_tangent;
    Vec2<DataT> uv0;
    Vec2<DataT> uv1;
    CUDAMaterial<DataT> material;
    uint32_t inscene_objid;
    uint32_t inobject_offset;
    RGBColor<DataT> vertex_color;
    uint32_t objectid;

    if constexpr (FIRST_ROUND) {
        uchar4 color_u4 = read_surface2d<uchar4>(input.gbuffer_color, x, y);
        vertex_color = to_color_float<DataT>(color_u4);
        bool empty = color_u4.w == 0;
        if (empty) {
            for (int i = 0; i < MAX_DIRECT_LIGHT; i++) {
                buffer.trace_light_commands[thread_id * MAX_DIRECT_LIGHT + i].slot = UINT8_MAX;
            }

            if (input.skybox.valid) {
                auto dir = settings.get_primary_ray_direction(x, y, true);
                auto tex_uv = direction_to_spherical(dir, input.skybox.delta_x, input.skybox.delta_y);
                float4 skybox_color = tex2D<float4>(input.skybox.texture, tex_uv[0], tex_uv[1]);
                buffer.di_intensity[thread_id * MAX_DIRECT_LIGHT + 0] =
                    RGBColor<DataT>(skybox_color.x, skybox_color.y, skybox_color.z) * input.skybox.exposure;
                // buffer.di_intensity[thread_id * MAX_DIRECT_LIGHT + 0] =
                //    RGBColor<DataT>(1, 0, 0);
            }
            return;
        }

        // 从 buffer 中读取数据
        inscene_objid = read_surface2d<uint32_t>(input.gbuffer_inscene_objid, x, y);
        inobject_offset = read_surface2d<uint32_t>(input.gbuffer_inobject_offset, x, y);

        Vec4<DataT> position_meaningless = read_surface2d<Vec4<DataT>>(input.gbuffer_position, x, y);
        position = position_meaningless.xyz();

        Vec4<DataT> normal_depth = read_surface2d<Vec4<DataT>>(input.gbuffer_normal_depth, x, y);
        raw_normal = normal_depth.xyz();

        Vec4<DataT> tangent_meaningless = read_surface2d<Vec4<DataT>>(input.gbuffer_tangent, x, y);
        raw_tangent = tangent_meaningless.xyz();

        Vec4<DataT> uv0uv1 = read_surface2d<Vec4<DataT>>(input.gbuffer_uv0uv1, x, y);
        uv0 = Vec2<DataT>(uv0uv1[0], uv0uv1[1]);
        uv1 = Vec2<DataT>(uv0uv1[2], uv0uv1[3]);
        material = input.material_constants[input.objects_constant[inscene_objid].materialid];
        objectid = input.objects_constant[inscene_objid].objectid;
    } else {
        // 从 shade command 来
        auto &shade_command = buffer.shade_commands[thread_id];
        if (buffer.shade_commands[thread_id].type == PixelShaderInput<DataT>::INVALID) {
            for (int i = 0; i < MAX_DIRECT_LIGHT; i++) {
                buffer.trace_light_commands[thread_id * MAX_DIRECT_LIGHT + i].slot = UINT8_MAX;
            }
            return;
        } else if (buffer.shade_commands[thread_id].type == PixelShaderInput<DataT>::SKYBOX) {
            for (int i = 0; i < MAX_DIRECT_LIGHT; i++) {
                buffer.trace_light_commands[thread_id * MAX_DIRECT_LIGHT + i].slot = UINT8_MAX;
            }
            auto dir = -buffer.view_direction[thread_id].normalized();
            auto tex_uv = direction_to_spherical(dir, input.skybox.delta_x, input.skybox.delta_y);
            float4 skybox_color = tex2D<float4>(input.skybox.texture, tex_uv[0], tex_uv[1]);
            // buffer.di_intensity[thread_id * MAX_DIRECT_LIGHT + 0] = SKYBOX_COLOR;
            buffer.di_intensity[thread_id * MAX_DIRECT_LIGHT + 0] = RGBColor<DataT>(DataT(skybox_color.x), DataT(skybox_color.y), DataT(skybox_color.z)) * input.skybox.exposure;
            return;
        } else {
            vertex_color = shade_command.color;
            inscene_objid = shade_command.inscene_objid;
            inobject_offset = shade_command.inobject_offset;
            position = shade_command.position;
            raw_normal = shade_command.normal;
            raw_tangent = shade_command.tangent;
            uv0 = shade_command.uv0;
            uv1 = shade_command.uv1;
            material = input.material_constants[shade_command.materialid];
            objectid = input.objects_constant[inscene_objid].objectid;
        }
    }

// 预处理：颜色
#define UV(uv) float((uv) == 0 ? uv0[0] : uv1[0]), float((uv) == 0 ? uv0[1] : uv1[1])
    RGBColor<DataT> color = material.color;
    if (material.uv_color < 2) {
        float4 tex_color = tex2D<float4>(material.tex_color, UV(material.uv_color));
        color = float4_to_vec4<DataT>(tex_color).xyz();
    }
    color = elemul(color, vertex_color);
#undef UV

    // 预处理：N, V
    Vec3<DataT> normal = raw_normal;

    // 如果 view 方向和 normal 方向不一致，并且是双面材质，那么翻转 normal
    Vec3<DataT> view_dir;
    if constexpr (FIRST_ROUND) {
        view_dir = -settings.get_primary_ray_direction(x, y, true);
    } else {
        view_dir = buffer.view_direction[thread_id];
    }

    if (dot_product(view_dir, normal) < DataT(0)) {
        if (material.double_sided) {
            normal = -normal;
        } else {
            // 不应当产生相交
            for (int i = 0; i < MAX_DIRECT_LIGHT; i++) {
                buffer.trace_light_commands[thread_id * MAX_DIRECT_LIGHT + i].slot = UINT8_MAX;
            }
            return;
        }
    }

    // TODO: 读取 normal map, 计算 normal map 后的 normal、tangent、bitangent
    // T x B = N, N x T = B
    auto raw_bitangent = cross_product(raw_normal, raw_tangent).normalized();
    // non-standard
    raw_tangent = cross_product(raw_bitangent, raw_normal).normalized();
    auto tangent = raw_tangent;
    auto bitangent = raw_bitangent;

    typename CUDARenderGIInput<DataT>::SkipGeometry skip{objectid, inobject_offset};
    buffer.skips[thread_id] = skip;

    // 自发光
    buffer.intensity[round * settings.width * settings.height + thread_id] = material.emission;

    if constexpr (FIRST_ROUND) {
        buffer.albedo[thread_id] = color;
    }

    // 处理全局光照
    if constexpr (!NO_GI) {
        TraceGICommand<DataT> &cmd = buffer.trace_gi_command[thread_id];
        cmd.x = x;
        cmd.y = y;
        cmd.source = position;
        if (material.roughness < DataT(0.1)) {
            if (DataT(curand_uniform(&rand_state)) < material.metallic) {
                cmd.direction = view_dir.symmetric_vector(normal).normalized();
                buffer.gi_material_multiplier[round * settings.width * settings.height + thread_id] =
                    glassy_brdf(material.metallic, view_dir, cmd.direction, normal).get_brdf(color) /
                    material.metallic;
                buffer.view_direction[thread_id] = -cmd.direction;
            } else {
                if (DataT(curand_uniform(&rand_state)) < DataT(0.6)) {
                    cmd.direction = view_dir.symmetric_vector(normal).normalized();
                    auto brdf = glassy_brdf(material.metallic, view_dir, cmd.direction, normal);
                    auto pdf = (DataT(1) - material.metallic) * DataT(0.6);
                    if constexpr (FIRST_ROUND) {
                        buffer.gi_material_multiplier[thread_id] =
                            RGBColor<DataT>(brdf.colored / pdf, brdf.white / pdf, nan<DataT>());
                    } else {
                        buffer.gi_material_multiplier[round * settings.width * settings.height + thread_id] =
                            brdf.get_brdf(color) / pdf;
                    }
                    buffer.view_direction[thread_id] = -cmd.direction;
                } else {
                    // 应当是 cosine weighted
                    Vec3<DataT> reflect_dir(curand_normal(&rand_state), curand_normal(&rand_state),
                                            curand_normal(&rand_state));
                    reflect_dir.normalize();
                    DataT cosine;
                    if ((cosine = dot_product(reflect_dir, normal)) < DataT(0)) {
                        reflect_dir = -reflect_dir;
                        cosine = -cosine;
                    }
                    cmd.direction = reflect_dir;
                    buffer.view_direction[thread_id] = -reflect_dir;
                    BRDF<DataT> brdf =
                        material_brdf(material.metallic, material.roughness, view_dir, reflect_dir, normal);
                    DataT multiplier = cosine * DataT(M_PI * 2);
                    if constexpr (FIRST_ROUND) {
                        buffer.gi_material_multiplier[thread_id] =
                            RGBColor<DataT>(brdf.colored * multiplier, brdf.white * multiplier, nan<DataT>());
                    } else {
                        // BRDF * COS * INTENSIT/ PDF
                        // Intensity  由 Trace 填写
                        buffer.gi_material_multiplier[round * settings.width * settings.height + thread_id] =
                            brdf.get_brdf(color) * multiplier;
                    }
                }
            }
        } else {
            DataT pdf;
            Vec3<DataT> reflect_dir;
            DataT cosine;
            if (DataT(curand_uniform(&rand_state)) < material.metallic) {
                auto a = material.roughness * material.roughness;
                auto a2 = a * a;
                auto half_dir_t = sample_ggx(a2, rand_state);
                pdf = pdf_ggx_reflect(half_dir_t[2], a2) * material.metallic;
                auto half_dir_w = tangent_to_world(half_dir_t, normal, tangent, bitangent).normalized();
                reflect_dir = view_dir.symmetric_vector(half_dir_w);
                cosine = max(DataT(0.05), dot_product(reflect_dir, half_dir_w));

                pdf /= (4 * cosine);

            } else {
                reflect_dir[0] = curand_normal(&rand_state);
                reflect_dir[1] = curand_normal(&rand_state);
                reflect_dir[2] = curand_normal(&rand_state);
                if ((cosine = dot_product(reflect_dir, normal)) < DataT(0)) {
                    reflect_dir = -reflect_dir;
                    cosine = -cosine;
                }
                pdf = DataT(0.5 / M_PI) * (DataT(1) - material.metallic);
            }
            reflect_dir.normalize();
            cmd.direction = reflect_dir;
            buffer.view_direction[thread_id] = -reflect_dir;

            BRDF<DataT> brdf(
                material_brdf(material.metallic, material.roughness, view_dir, reflect_dir, normal));
            /*    BRDF<DataT> brdf(
                    material_brdf((float)material.metallic, (float)material.roughness, Vec3f(view_dir),
               Vec3f(reflect_dir), Vec3f(normal)));*/

            DataT multiplier = cosine / pdf;
            if constexpr (FIRST_ROUND) {
                buffer.gi_material_multiplier[thread_id] =
                    RGBColor<DataT>(brdf.colored * multiplier, brdf.white * multiplier, nan<DataT>());
            } else {
                buffer.gi_material_multiplier[round * settings.width * settings.height + thread_id] =
                    multiplier * brdf.get_brdf(color);
            }
            // BRDF * COS * INTENSITY / R^2 / PDF
            // Intensity  R^2 由 Trace 填写
        }
    } else {
        buffer.trace_gi_command[thread_id].x = UINT16_MAX;
    }

    if constexpr (NO_GI) {
        // 如果没有 GI 的话，那么就做个简单的全局光
        // 假设能量守恒且0.5的方向都可以接收到天光
        buffer.intensity[round * settings.width * settings.height + thread_id] =
            elemul(SKYBOX_COLOR, color) * DataT(0.5);
    }

    // 处理直接光照
    for (int i = 0; i < MAX_DIRECT_LIGHT; i++) {

        // 预先填充初始值
        TraceLightCommand<DataT> &cmd = buffer.trace_light_commands[thread_id * MAX_DIRECT_LIGHT + i];
        cmd.x = x;
        cmd.y = y;
        cmd.source = position;
        cmd.slot = UINT8_MAX;

        // 如果光线数量不够，则 continue
        // continue 是为了填充初始值
        if (i >= input.n_lights)
            continue;

        RDLight<DataT> light = input.lights[i];
        if (light.type == RDLight<DataT>::LightType::POINT || light.type == RDLight<DataT>::LightType::SPOT) {
            Vec3<DataT> light_distance_vec = light.position - position;
            auto light_dir = light_distance_vec.normalized();

            // 如果光线方向不对，则 continue
            DataT cosine = dot_product(light_dir, normal);
            if (cosine < DataT(0))
                continue;

            DataT light_distance2 = light_distance_vec.norm2_squared();
            Vec3<DataT> brdf = material_brdf(material.metallic, max(material.roughness, DataT(0.10f)),
                                             view_dir, light_dir, normal)
                                   .get_brdf(color);
            cmd.direction = light_dir;
            cmd.slot = i;
            cmd.material_multiplier = (cosine / light_distance2 / DataT(10)) * elemul(brdf, light.intensity);
            cmd.maximum_t = sqrt(light_distance2);
        } else {
            Vec3<DataT> light_dir = -light.direction.normalized();
            DataT cosine = dot_product(light_dir, normal);
            if (cosine < DataT(0))
                continue;

            Vec3<DataT> brdf = material_brdf(material.metallic, max(material.roughness, DataT(0.10f)),
                                             view_dir, light_dir, normal)
                                   .get_brdf(color);
            cmd.direction = light_dir;
            cmd.slot = i;
            cmd.material_multiplier = cosine * elemul(brdf, light.intensity);
            cmd.maximum_t = 1000;
        }
    }
}

template <typename DataT, size_t MAX_DIRECT_LIGHT>
__global__ void trace_di_light(CUDARenderGIInput<DataT> input,
                               typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer, bool no_gi,
                               CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_id >= settings.width * settings.height * MAX_DIRECT_LIGHT)
        return;

    TraceLightCommand<DataT> command = buffer.trace_light_commands[thread_id];
    if (command.slot == UINT8_MAX) {
        //    printf("skiped\n");
        return;
    }

    auto shade_tid = command.y * settings.width + command.x;

    typename CUDARenderGIInput<DataT>::SkipGeometry skip = buffer.skips[shade_tid];
    Intersection<DataT> intersection(input.cast_ray<true, false>(
        Ray<DataT>{command.source, command.direction}, skip, RAY_MOVEFORWARD_T<DataT>(), command.maximum_t));

    buffer.di_intensity[MAX_DIRECT_LIGHT * shade_tid + command.slot] =
        (intersection.intersected() ? DataT(0) : DataT(1)) * command.material_multiplier;
}

template <typename DataT, size_t MAX_DIRECT_LIGHT>
__global__ void trace_gi(CUDARenderGIInput<DataT> input,
                         typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer, int round,
                         CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_id >= settings.width * settings.height)
        return;

    TraceGICommand<DataT> command = buffer.trace_gi_command[thread_id];

    if (command.x == UINT16_MAX) {
        return;
    }

    auto shade_tid = command.y * settings.width + command.x;

    typename CUDARenderGIInput<DataT>::SkipGeometry skip = buffer.skips[shade_tid];
    Intersection<DataT> intersection(input.cast_ray<false, false>(
        Ray<DataT>{command.source, command.direction}, skip, RAY_MOVEFORWARD_T<DataT>()));

    auto &shade_command = buffer.shade_commands[shade_tid];

    if (intersection.intersected()) {
        auto vao_index = input.objects_constant[intersection.inscene_objid].objectid;
        const CUDAMeshVAO vao = input.vaos[vao_index];
        const RDVertex<DataT> *vbo = input.vbos[vao.vbo_index];
        const uint32_t *ebo = input.ebos[vao.ebo_index];

        RDVertex<DataT> v1 = vbo[ebo[vao.ebo_offset + intersection.inobject_offset + 0]];
        RDVertex<DataT> v2 = vbo[ebo[vao.ebo_offset + intersection.inobject_offset + 1]];
        RDVertex<DataT> v3 = vbo[ebo[vao.ebo_offset + intersection.inobject_offset + 2]];

        auto position = lerp(intersection.barycentric, v1.position, v2.position, v3.position);
        auto normal = lerp(intersection.barycentric, v1.normal, v2.normal, v3.normal).normalized();
        auto tangent = lerp(intersection.barycentric, v1.tangent, v2.tangent, v3.tangent).normalized();

        auto transform = input.objects_constant[intersection.inscene_objid].transform_L2W;

        shade_command.normal = (transform * normal.to_vec4_as_dir()).to_vec3_as_dir().normalized();
        shade_command.tangent = (transform * tangent.to_vec4_as_dir()).to_vec3_as_dir().normalized();
        shade_command.position = (transform * position.to_vec4_as_pos()).to_vec3_as_pos();

        shade_command.color = lerp(intersection.barycentric, v1.color, v2.color, v3.color);
        shade_command.uv0 = lerp(intersection.barycentric, v1.uv0, v2.uv0, v3.uv0);
        shade_command.uv1 = lerp(intersection.barycentric, v1.uv1, v2.uv1, v3.uv1);

        shade_command.inscene_objid = intersection.inscene_objid;
        shade_command.inobject_offset = intersection.inobject_offset;
        shade_command.materialid = input.objects_constant[intersection.inscene_objid].materialid;

        shade_command.type = PixelShaderInput<DataT>::COMMON;
    } else {
        // 如果有设置天空盒，可以在此处采样天空盒

        shade_command.type = PixelShaderInput<DataT>::SKYBOX;
    }
    return;
}

template <typename DataT, size_t MAX_DIRECT_LIGHT>
__global__ void accumulate_di_light(typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                    int round, CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    RGBColor<DataT> final_color = {};
    for (int i = 0; i < MAX_DIRECT_LIGHT; i++) {
        final_color += buffer.di_intensity[thread_id * MAX_DIRECT_LIGHT + i];
    }

    buffer.intensity[settings.width * settings.height * round + thread_id] += final_color;
}

struct DemoSetting {
    bool add_direct_out = true;
    bool add_gi_colored = true;
    bool add_gi_white = true;
    bool demodulate = false;

    bool svgf = true;
};

template <typename DataT, size_t MAX_DIRECT_LIGHT>
__global__ void write_clean_color(CUDARenderGIInput<DataT> input,
                                  typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                  int n_rounds, CUDARenderGISettings<DataT> settings,
                                  DemoSetting demo_setting) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    auto multiplier0 = buffer.gi_material_multiplier[0 * settings.width * settings.height + thread_id];
    RGBColor<DataT> final_color;
    if (demo_setting.add_direct_out) {
        final_color = buffer.intensity[0 * settings.width * settings.height + thread_id];
    }
    auto intensity1 = buffer.intensity[1 * settings.width * settings.height + thread_id];

    if (isnan(multiplier0[2])) {
        buffer.mul_gi_colored[thread_id] = multiplier0[0] * intensity1;
        buffer.mul_gi_white[thread_id] = multiplier0[1] * intensity1;
    } else {
        if (demo_setting.add_direct_out) {
            final_color += elemul(intensity1, multiplier0);
        }
    }

    buffer.color_inprogress[thread_id] = final_color;
}

template <typename DataT, size_t MAX_DIRECT_LIGHT>
__global__ void add_denoised_color(cudaSurfaceObject_t out, CUDARenderGIInput<DataT> input,
                                   typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                   int n_rounds, CUDARenderGISettings<DataT> settings,
                                   DemoSetting demo_setting) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    RGBColor<DataT> albedo = demo_setting.demodulate ? RGBColor<DataT>(1, 1, 1) : buffer.albedo[thread_id];

    auto& color = buffer.color_inprogress[thread_id];

    if (demo_setting.add_gi_colored) {
        color += elemul(buffer.mul_gi_colored[thread_id], albedo);
    }
    if (demo_setting.add_gi_white) {
        color += buffer.mul_gi_white[thread_id];
    }

    //RGBAColorF color_out;
    //color_out.xyz() = elepow(RGBColorF(color), 1 / 2.2f);
    //color_out[3] = 1;
    //write_surface2d(color_out, out, x, y);
}

template <typename DataT, uint32_t MAX_DIRECT_LIGHT>
__global__ void copy_last_frame_pixel_id(cudaSurfaceObject_t out, CUDARenderGIInput<DataT> input,
                                         typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                         CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    if (read_surface2d<uchar4>(input.gbuffer_color, x, y).w == 0) {
        buffer.last_frame_id[thread_id].object_id = UINT32_MAX;
    } else {
        buffer.last_frame_id[thread_id].object_id =
            input.objects_constant[read_surface2d<uint32_t>(input.gbuffer_inscene_objid, x, y)].objectid;
        buffer.last_frame_id[thread_id].inobject_offset =
            read_surface2d<uint32_t>(input.gbuffer_inobject_offset, x, y);
    }

    //    buffer.color_inprogress[thread_id] = read_surface2d<RGBAColor<DataT>>(out, x, y).xyz().clone();
}

template <typename DataT, uint32_t MAX_DIRECT_LIGHT>
__global__ void copy_last_frame_transform(CUDARenderGIInput<DataT> input,
                                          typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                          uint32_t count) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= count)
        return;

    auto constant = input.objects_constant[thread_id];
    buffer.last_frame_L2W[constant.objectid] = constant.transform_L2W;
}

template <typename DataT, uint32_t MAX_DIRECT_LIGHT>
__global__ void generate_temporal_map_step1(CUDARenderGIInput<DataT> input,
                                            typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                            CUDARenderGISettings<DataT> settings, long random_seed) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    auto empty = read_surface2d<uchar4>(input.gbuffer_color, x, y).w == 0;
    if (empty) {
        buffer.svgf_temporal_map[thread_id].frame_count = 0;
        return;
    }
    auto object_constant =
        input.objects_constant[read_surface2d<uint32_t>(input.gbuffer_inscene_objid, x, y)];
    auto inobject_offset = read_surface2d<uint32_t>(input.gbuffer_inobject_offset, x, y);
    auto coordinate_W4 = read_surface2d<Vec4<DataT>>(input.gbuffer_position, x, y);
    coordinate_W4[3] = 1;
    auto coordinate_L4 = object_constant.transform_W2L * coordinate_W4;
    auto coordinate_W4_lf = buffer.last_frame_L2W[object_constant.objectid] * coordinate_L4;
    auto coordinate_C4 = buffer.last_frame_transform_W2C * coordinate_W4_lf;
    DataT g_fx = (DataT(1) + coordinate_C4[0] / coordinate_C4[3]) / DataT(2) * DataT(settings.width);
    DataT g_fy = (DataT(1) + coordinate_C4[1] / coordinate_C4[3]) / DataT(2) * DataT(settings.height);

    // fx \in [0, settings.width], fy \in [0, settings.height]

    // svgf 2x2
    {
        for (int i = 0; i < 4; i++) {
            buffer.svgf_temporal_map[thread_id].weights[i] = 0;
        }

        DataT fx = g_fx - DataT(0.5);
        DataT fy = g_fy - DataT(0.5);
        int lx = (int)(fx);
        int ly = (int)(fy);
        int ux = lx + 1;
        int uy = ly + 1;

        int count = 0;

        Vec2<int> nxnys[4] = {Vec2<int>(lx, ly), Vec2<int>(lx, uy), Vec2<int>(ux, ly), Vec2<int>(ux, uy)};
        DataT weights[4] = {(DataT(ux) - fx) * (DataT(uy) - fy), (DataT(ux) - fx) * (fy - DataT(ly)),
                            (fx - DataT(lx)) * (DataT(uy) - fy), (fx - DataT(lx)) * (fy - DataT(ly))};
        DataT total_weight = 0;
        DataT total_weight_taa = 0;
        for (int i = 0; i < 4; i++) {
            auto nxny = nxnys[i];
            auto weight = weights[i];

            auto nx = nxny[0];
            auto ny = nxny[1];

            if (nx >= 0 && nx < settings.width && ny >= 0 && ny < settings.height) {
                auto tid = ny * settings.width + nx;
                auto lid = buffer.last_frame_id[tid];

                if (/* lid.inobject_offset == inobject_offset && */
                    lid.object_id == object_constant.objectid) {
                    count = max<int>(count, buffer.svgf_temporal_map[tid].frame_count);
                    buffer.svgf_temporal_map[thread_id].weights[i] = weights[i];
                    total_weight += weights[i];
                    buffer.svgf_temporal_map[thread_id].last_frame_pos[i] = Vec2<uint16_t>(nx, ny);
                }
            }
        }

        if (total_weight == DataT(0)) {
            buffer.valid_frame_count[thread_id] = 0;
        } else {
            for (int i = 0; i < 4; i++) {
                if (count < 255)
                    count += 1;
                buffer.svgf_temporal_map[thread_id].weights[i] /= total_weight;
            }
            buffer.valid_frame_count[thread_id] = count;
        }
    }

    // taa 2x2
    {
        for (int i = 0; i < 4; i++) {
            buffer.taa_temporal_map[thread_id].weights[i] = 0;
        }

        curandState state;
        curand_init(random_seed, thread_id, 0, &state);

        DataT fx = g_fx - DataT(curand_uniform(&state));
        DataT fy = g_fy - DataT(curand_uniform(&state));
        int lx = (int)(fx);
        int ly = (int)(fy);
        int ux = lx + 1;
        int uy = ly + 1;
        
        Vec2<int> nxnys[4] = {Vec2<int>(lx, ly), Vec2<int>(lx, uy), Vec2<int>(ux, ly), Vec2<int>(ux, uy)};
        DataT weights[4] = {(DataT(ux) - fx) * (DataT(uy) - fy), (DataT(ux) - fx) * (fy - DataT(ly)),
                            (fx - DataT(lx)) * (DataT(uy) - fy), (fx - DataT(lx)) * (fy - DataT(ly))};
        DataT total_weight = 0;
        DataT total_weight_taa = 0;
        bool same_obj_occurred = false;
        bool other_obj_occurred = false;
        for (int i = 0; i < 4; i++) {
            auto nxny = nxnys[i];
            auto weight = weights[i];

            auto nx = nxny[0];
            auto ny = nxny[1];

            if (nx >= 0 && nx < settings.width && ny >= 0 && ny < settings.height) {
                auto tid = ny * settings.width + nx;
                auto lid = buffer.last_frame_id[tid];
                
                buffer.taa_temporal_map[thread_id].weights[i] = weights[i];
                total_weight += weights[i];
                buffer.taa_temporal_map[thread_id].last_frame_pos[i] = Vec2<uint16_t>(nx, ny);

                if (lid.object_id == object_constant.objectid) {
                    same_obj_occurred = true;
                } else {
                    other_obj_occurred = true;
                }
            }
        }

        if (!same_obj_occurred) {
            buffer.taa_temporal_map[thread_id].frame_count = 0;
        } else {
            for (int i = 0; i < 4; i++) {
                buffer.taa_temporal_map[thread_id].weights[i] /= total_weight;
            }
            buffer.taa_temporal_map[thread_id].frame_count = 1;
        }
    }

    // TODO
    // try2: 3x3
}

template <typename DataT, uint32_t MAX_DIRECT_LIGHT>
__global__ void generate_temporal_map_step2(CUDARenderGIInput<DataT> input,
                                            typename RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>::Ref buffer,
                                            CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    buffer.svgf_temporal_map[thread_id].frame_count = buffer.valid_frame_count[thread_id];
}

template <typename DataT> __global__ void temporal_anti_aliasing(
    RGBColor<DataT>* color_inout, 
    RGBColor<DataT>* history,
    TemporalMap<DataT>* temporal,
    CUDARenderGISettings<DataT> settings,
    DataT taa_weight
) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

    uint32_t x = thread_id % settings.width;
    uint32_t y = thread_id / settings.width;

    RGBColor<DataT> in_color = color_inout[thread_id];
    RGBColor<DataT> out_color = sample_linear<DataT, RGBColor<DataT>, DataT, RGBColor<DataT> *>(
        temporal[thread_id], history, settings.width, settings.height, color_inout[thread_id]);

    value_or(out_color, in_color);

    color_inout[thread_id] = out_color * (DataT(1) - taa_weight) + in_color * taa_weight;
}

template <typename DataT>
__global__ void write_to_surface2d(cudaSurfaceObject_t out, RGBColor<DataT> *input,
                                   CUDARenderGISettings<DataT> settings) {
    auto thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id >= settings.width * settings.height)
        return;

     uint32_t x = thread_id % settings.width;
     uint32_t y = thread_id / settings.width;

     RGBAColorF color_out;
     color_out.xyz() = elepow(RGBColorF(input[thread_id]), 1 / 2.2f);
     color_out[3] = 1;
     write_surface2d(color_out, out, x, y);
}

} // namespace rt

#endif