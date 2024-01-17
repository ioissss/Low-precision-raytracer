#include <cuda_fp16.h>
#include <half.hpp>

#ifndef RT_MATH_NUMBER_HPP
#define RT_MATH_NUMBER_HPP

struct float16 {
  private:
    static half_float::half &cpu(__half &val) { return reinterpret_cast<half_float::half &>(val); }
    static half_float::half cpu_c(const __half val) { return reinterpret_cast<const half_float::half &>(val); }
    static __half& cuda(half_float::half& val) { return reinterpret_cast<__half &>(val); }

  public:
    __half data;
    float16() = default;
    __device__ __host__ float16(float val) : data(val) {}
    __host__ float16(half_float::half val) : data(cuda(val)) {}
    __device__ __host__ operator float() const { return float{data}; }

    __device__ __host__ float16 &operator+=(float16 other) {
#ifdef __CUDA_ARCH__
        data += other.data;
#else
        cpu(data) += cpu(other.data);
#endif
        return *this;
    }

    __device__ __host__ float16 &operator-=(float16 other) {
#ifdef __CUDA_ARCH__
        data -= other.data;
#else
        cpu(data) -= cpu(other.data);
#endif
        return *this;
    }

    __device__ __host__ float16 &operator*=(float16 other) {
#ifdef __CUDA_ARCH__
        data *= other.data;
#else
        cpu(data) *= cpu(other.data);
#endif
        return *this;
    }

    __device__ __host__ float16 &operator/=(float16 other) {
#ifdef __CUDA_ARCH__
        data /= other.data;
#else
        cpu(data) /= cpu(other.data);
#endif
        return *this;
    }

    __device__ __host__ float16 operator+(float16 other) const {
#ifdef __CUDA_ARCH__
        return {data + other.data};
#else
        return {cpu_c(data) + cpu(other.data)};
#endif
    }

    __device__ __host__ float16 operator-(float16 other) const {
#ifdef __CUDA_ARCH__
        return {data - other.data};
#else
        return {cpu_c(data) - cpu(other.data)};
#endif
    }

    __device__ __host__ float16 operator*(float16 other) const {
#ifdef __CUDA_ARCH__
        return {data * other.data};
#else
        return {cpu_c(data) * cpu(other.data)};
#endif
    }

    __device__ __host__ float16 operator/(float16 other) const {
#ifdef __CUDA_ARCH__
        return {data / other.data};
#else
        return {cpu_c(data) / cpu(other.data)};
#endif
    }

    __device__ __host__ bool operator<(float16 other) const {
#ifdef __CUDA_ARCH__
        return data < other.data;
#else
        return cpu_c(data) < cpu(other.data);
#endif
    }
    __device__ __host__ bool operator>(float16 other) const {
#ifdef __CUDA_ARCH__
        return data > other.data;
#else
        return cpu_c(data) > cpu(other.data);
#endif
    }
    __device__ __host__ bool operator<=(float16 other) const {
#ifdef __CUDA_ARCH__
        return data <= other.data;
#else
        return cpu_c(data) <= cpu(other.data);
#endif
    }
    __device__ __host__ bool operator>=(float16 other) const {
#ifdef __CUDA_ARCH__
        return data >= other.data;
#else
        return cpu_c(data) >= cpu(other.data);
#endif
    }
    __device__ __host__ bool operator==(float16 other) const {
#ifdef __CUDA_ARCH__
        return data == other.data;
#else
        return cpu_c(data) == cpu(other.data);
#endif
    }
    __device__ __host__ bool operator!=(float16 other) const {
#ifdef __CUDA_ARCH__
        return data != other.data;
#else
        return cpu_c(data) != cpu(other.data);
#endif
    }
};

inline __device__ __host__ float16 tan(float16 val) { return {tan(val.data)}; }
inline __device__ __host__ float16 pow(float16 val, float16 p) { return {pow(val.data, p.data)}; }
inline __device__ __host__ float16 abs(float16 val){
#ifdef __CUDA_ARCH__
    return {__habs(val.data)};
#else
    return {fabs(val.data)};
#endif
}
inline __device__ __host__ float16 sqrt(float16 val) { return {sqrt(val.data)}; }
inline __device__ __host__ bool isfinite(float16 val) {
#ifdef __CUDA_ARCH__
    return !(__hisinf(val.data) || __hisnan(val.data));
#else
    return isfinite(val.data);
#endif
}
inline __device__ __host__ bool isnan(float16 val) {
#ifdef __CUDA_ARCH__
    return __hisnan(val.data);
#else
    return isnan(val.data);
#endif
}

namespace rt {
template <typename T> __host__ __device__ T min(const T &a, const T &b) {
    return a < b ? a : b;
}
template <typename T> __host__ __device__ T max(const T &a, const T &b) {
    return a > b ? a : b;
}
template <typename T> __host__ __device__ T nan() { return {NAN}; }

} // namespace rt

#endif