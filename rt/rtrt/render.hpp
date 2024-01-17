#ifndef RT_RTRT_RENDER_HPP
#define RT_RTRT_RENDER_HPP

#include <type_traits>
#include <vector>
#include <memory>
#include "memory.hpp"
#include "math/number.hpp"
//
//namespace rt {
//
//template <typename DataT> struct RendererScene {
//
//    // 场景 AABB 存储为 float32，构建 BVH 后存储为 float16
//    std::vector<ObjectConstantData> objects_constants;
//    std::vector<RDLight<DataT>> lights;
//};
//
//// 资源类型，不能轻易改变
//template <typename DataT> struct RendererResource {
//    std::vector<GLMeshVAO> gl_vaos;
//    std::vector<std::pair<glm::vec3, glm::vec3>> objects_aabb;
//    std::vector<std::shared_ptr<ObjectBVH<DataT>>> objects_bvh;
//
//    
//    std::vector<std::shared_ptr<MeshEBO>> objects_ebo;
//    std::vector<std::shared_ptr<MeshVBOType>> objects_vbo;
//
//    std::shared_ptr<ObjectBVH<DataT>> objects_bvh;
//
//    std::vector<MaterialHolder<DataT>> materials;
//
//
//};
//
//template <typename DataT, bool IS_HALF = std::is_same_v<DataT, float16>,
//          bool IS_FLOAT = std::is_same_v<DataT, float>, typename = std::enable_if_t<IS_HALF || IS_FLOAT>>
//class PathTracingRenderer {};
//
//} // namespace rt

#endif // !1
