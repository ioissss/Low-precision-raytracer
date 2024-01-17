// 加载器，加载到泛型类型

#ifndef RTRT_LOADER_HPP
#define RTRT_LOADER_HPP

#include "hierarchy.hpp"
#include "memory.hpp"
#include <map>
#include <memory>
#include <type_traits>

namespace rt {


template <typename DataT>
std::shared_ptr<hierarchy::Object<DataT>> load_gltf2(const std::string &path, RDResource<DataT> &rd_resource);

} // namespace rt

#endif