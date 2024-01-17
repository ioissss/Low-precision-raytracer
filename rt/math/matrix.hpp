#ifndef MATRIX_HPP
#define MATRIX_HPP

#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923

#include "cmath"

#include "utils/log.hpp"

#include <functional>
#include <thrust/pair.h>

namespace rt {

template <int M, int N, typename DataT> class Matrix;

template <int M, int N, typename DataT> class MatrixSlice {
  public:
    DataT *data[M][N];

    __device__ __host__ DataT &operator[](const thrust::pair<int, int> &index) {
        return *data[index.first][index.second];
    }

    __device__ __host__ DataT operator[](const thrust::pair<int, int> &index) const {
        return *data[index.first][index.second];
    }

    __device__ __host__ MatrixSlice &operator=(const MatrixSlice &slice) {
        if (&slice == this)
            return *this;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                *data[i][j] = *slice.data[i][j];
            }
        }
        return *this;
    }

    __device__ __host__ MatrixSlice &operator=(const Matrix<M, N, DataT> &mat) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                *data[i][j] = mat.data[i][j];
            }
        }
        return *this;
    }

    __device__ __host__ MatrixSlice &operator*(const DataT num) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                *data[i][j] *= num;
            }
        }
        return *this;
    }

    __device__ __host__ MatrixSlice &operator-=(const MatrixSlice rhs) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                *data[i][j] -= *rhs.data[i][j];
            }
        }
        return *this;
    }

    __device__ __host__ MatrixSlice &operator/=(const DataT num) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                *data[i][j] /= num;
            }
        }
        return *this;
    }

    __device__ __host__ Matrix<M, N, DataT> clone() const;

    __device__ __host__ MatrixSlice<N, M, DataT> transposed() {
        MatrixSlice<N, M, DataT> out;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                out.data[j][i] = data[i][j];
        return out;
    }

    __device__ __host__ MatrixSlice<1, N, DataT> row(int i) {
        MatrixSlice<1, N, DataT> out;
        for (int j = 0; j < N; j++) {
            out.data[0][j] = data[i][j]; // TODO(wangweihao): ????
        }
        return out;
    }

    __device__ __host__ MatrixSlice<M, 1, DataT> col(int j) {
        MatrixSlice<M, 1, DataT> out;
        for (int i = 0; i < M; i++) {
            out.data[i][0] = data[i][j];
        }
        return out;
    }

    MatrixSlice() = default;
    MatrixSlice(const MatrixSlice &) = default;
};

template <int M, int N, typename DataT>
void swap_slice_data(const rt::MatrixSlice<M, N, DataT> slice1, const rt::MatrixSlice<M, N, DataT> &slice2) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::swap(*slice1.data[i][j], *slice1.data[i][j]);
        }
    }
}

template <int M, int N, typename DataT> class Matrix {

  public:
    DataT data[M][N] = {};
    using InnerDataType = DataT;

    static constexpr int ShapeM = M;
    static constexpr int ShapeN = N;

    Matrix() = default;

    Matrix(const Matrix &other) = default;

    __device__ __host__ Matrix(const MatrixSlice<M, N, DataT> &other) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] = *other.data[i][j];
            }
        }
    }

  protected:
    template <int INDEX, typename DataT1, typename = std::enable_if_t<std::is_convertible_v<DataT1, DataT>>>
    __device__ __host__ void init(DataT *ptr, const DataT1 &val) {
        *ptr = DataT(val);
    }

    template <int INDEX, typename DataT1, typename... DataTs,
              typename = std::enable_if_t<std::is_convertible_v<DataT1, DataT>>>
    __device__ __host__ void init(DataT *ptr, const DataT1 &val, const DataTs &...args) {
        *ptr = DataT(val);
        init<INDEX + 1, DataTs...>(ptr + 1, args...);
    }

  public:
    template <typename... DataTs, typename = std::enable_if_t<sizeof...(DataTs) == M * N>>
    __device__ __host__ Matrix(const DataTs &...args) {
        init<0, DataTs...>((DataT *)&data, args...);
    }

    template <bool IS_VECTOR = N == 1, typename = std::enable_if_t<IS_VECTOR>>
    Matrix(const std::initializer_list<DataT> &init) {
        int i = 0;
        for (auto val : init) {
            data[i++][0] = val;
        }
    }

    Matrix(const std::initializer_list<std::initializer_list<DataT>> &init) {
        int i = 0, j = 0;
        for (auto row : init) {
            for (auto val : row) {
                data[i][j++] = val;
            }
            i++;
            j = 0;
        }
    }

    __device__ __host__ bool operator==(const Matrix &other) const {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (data[i][j] != other.data[i][j])
                    return false;
        return true;
    }

    __device__ __host__ DataT &operator[](const thrust::pair<int, int> &index) {
        return data[index.first][index.second];
    }

    __device__ __host__ DataT operator[](const thrust::pair<int, int> &index) const {
        return data[index.first][index.second];
    }

    template <bool IS_VECTOR = N == 1>
    __device__ __host__ std::enable_if_t<IS_VECTOR, DataT &> operator[](int index) {
        return data[index][0];
    }

    template <bool IS_VECTOR = N == 1>
    __device__ __host__ std::enable_if_t<IS_VECTOR, DataT> operator[](int index) const {
        return data[index][0];
    }

    __device__ __host__ Matrix<N, M, DataT> transposed() {
        Matrix<N, M, DataT> out;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                out[{j, i}] = data[i][j];
        return out;
    }

    template <int O> __device__ __host__ Matrix<N, O, DataT> operator*(const Matrix<N, O, DataT> &rhs) const {
        Matrix<M, O, DataT> mat;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < O; k++) {
                    mat[{i, k}] += data[i][j] * rhs[{j, k}];
                }
            }
        }
        return mat;
    }

    __device__ __host__ Matrix &operator*=(DataT k) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] *= k;
            }
        }
        return *this;
    }

    __device__ __host__ Matrix operator*(DataT k) const {
        Matrix rst = *this;
        rst *= k;
        return rst;
    }

    __device__ __host__ Matrix &operator+=(const Matrix &rhs) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] += rhs.data[i][j];
            }
        }
        return *this;
    }

    __device__ __host__ Matrix operator+(const Matrix &rhs) const {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ Matrix &operator+=(DataT k) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] += k;
            }
        }
        return *this;
    }

    __device__ __host__ Matrix &operator-=(DataT k) { return *this += -k; }

    __device__ __host__ Matrix operator+(DataT rhs) const {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ Matrix operator-(DataT rhs) const {
        auto lhs = *this;
        return lhs -= rhs;
    }

    __device__ __host__ Matrix operator-(const Matrix &rhs) const {
        Matrix out = *this;
        return out -= rhs;
    }

    __device__ __host__ Matrix &operator-=(const Matrix &rhs) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] -= rhs.data[i][j];
            }
        }
        return *this;
    }

    template <bool IS_SUQARE = M == N>
    __device__ __host__ static std::enable_if_t<IS_SUQARE, Matrix> identity() {
        Matrix out{};
        for (int i = 0; i < M; i++) {
            out[{i, i}] = 1;
        }
        return out;
    };

    template <bool IS_SQUARE = M == N>
    static std::enable_if_t<IS_SQUARE, Matrix> diag(const std::initializer_list<DataT> &init) {
        Matrix out;
        int i = 0;
        for (auto n : init) {
            out[{i, i}] = n;
            i++;
        }
        return out;
    };

    __device__ __host__ DataT norm2_squared() const {
        DataT sum(0);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                sum += data[i][j] * data[i][j];
            }
        }
        return sum;
    }

    __device__ __host__ DataT norm2() const { return sqrt(norm2_squared()); }

    __device__ __host__ Matrix &operator/=(DataT x) { return (*this) *= 1.0 / x; }

    __device__ __host__ Matrix operator/(DataT x) {
        auto out = *this;
        out /= x;
        return out;
    }

    template <bool IS_VECTOR = N == 1> __device__ __host__ std::enable_if_t<IS_VECTOR, Matrix> &normalize() {
        return (*this) /= norm2();
    }

    template <bool IS_VECTOR = N == 1>
    __device__ __host__ std::enable_if_t<IS_VECTOR, Matrix> normalized() const {
        auto out = *this;
        return out /= norm2();
    }

    template <bool IS_SQUARE = N == M>
    __device__ __host__ std::enable_if_t<IS_SQUARE, Matrix> inversed2(bool &bCanReverse) const {
        int i, j, k;
        DataT W[N][2 * N]; //, result[N][N];
        DataT tem_1, tem_2, tem_3;
        Matrix re;

        // 对矩阵右半部分进行扩增
        for (i = 0; i < N; i++) {
            for (j = 0; j < 2 * N; j++) {
                if (j < N) {
                    W[i][j] = data[i][j];
                } else {
                    W[i][j] = (j - N == i ? 1 : 0);
                }
            }
        }

        for (i = 0; i < N; i++) {
            // 判断矩阵第一行第一列的元素是否为0，若为0，继续判断第二行第一列元素，直到不为0，将其加到第一行
            if (W[i][i] == 0) {
                for (j = i + 1; j < N; j++) {
                    if (W[j][i] != 0)
                        break;
                }
                if (j == N) {
                    bCanReverse = false;
                    return re;
                    // break;
                }
                //将前面为0的行加上后面某一行
                for (k = 0; k < 2 * N; k++) {
                    W[i][k] += W[j][k];
                }
            }

            //将前面行首位元素置1
            tem_1 = W[i][i];
            for (j = 0; j < 2 * N; j++) {
                W[i][j] = W[i][j] / tem_1;
            }

            //将后面所有行首位元素置为0
            for (j = i + 1; j < N; j++) {
                tem_2 = W[j][i];
                for (k = i; k < 2 * N; k++) {
                    W[j][k] = W[j][k] - tem_2 * W[i][k];
                }
            }
        }

        // 将矩阵前半部分标准化
        for (i = N - 1; i >= 0; i--) {
            for (j = i - 1; j >= 0; j--) {
                tem_3 = W[j][i];
                for (k = i; k < 2 * N; k++) {
                    W[j][k] = W[j][k] - tem_3 * W[i][k];
                }
            }
        }

        //得出逆矩阵
        for (i = 0; i < N; i++) {
            for (j = N; j < 2 * N; j++) {
                re.data[i][j - N] = W[i][j];
            }
        }

        return re;
    }

    template <bool IS_SQUARE = N == M>
    __device__ __host__ std::enable_if_t<IS_SQUARE, Matrix> inversed_3_3() {
        Matrix<3, 3, float> minv; // inverse of matrix m
        float det = data[0][0] * DifferenceOfProducts(data[1][1], data[2][2], data[2][1], data[1][2]) -
                    data[0][1] * DifferenceOfProducts(data[1][0], data[2][2], data[1][2], data[2][0]) +
                    data[0][2] * DifferenceOfProducts(data[1][0], data[2][1], data[1][1], data[2][0]);

        float invdet = 1 / det;

        minv.data[0][0] = DifferenceOfProducts(data[1][1], data[2][2], data[2][1], data[1][2]) * invdet;
        minv.data[0][1] = DifferenceOfProducts(data[0][2], data[2][1], data[0][1], data[2][2]) * invdet;
        minv.data[0][2] = DifferenceOfProducts(data[0][1], data[1][2], data[0][2], data[1][1]) * invdet;
        minv.data[1][0] = DifferenceOfProducts(data[1][2], data[2][0], data[1][0], data[2][2]) * invdet;
        minv.data[1][1] = DifferenceOfProducts(data[0][0], data[2][2], data[0][2], data[2][0]) * invdet;
        minv.data[1][2] = DifferenceOfProducts(data[1][0], data[0][2], data[0][0], data[1][2]) * invdet;
        minv.data[2][0] = DifferenceOfProducts(data[1][0], data[2][1], data[2][0], data[1][1]) * invdet;
        minv.data[2][1] = DifferenceOfProducts(data[2][0], data[0][1], data[0][0], data[2][1]) * invdet;
        minv.data[2][2] = DifferenceOfProducts(data[0][0], data[1][1], data[1][0], data[0][1]) * invdet;

        return minv;
    }

    template <bool IS_SQUARE = N == M>
    __device__ __host__ std::enable_if_t<IS_SQUARE, Matrix> inversed_4_4(bool &bCanReverse) {
        //求解4*4矩阵的逆
        Matrix re;
        DataT m[16];
        int k = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[k] = data[i][j];
                k++;
            }
        }

        double inv[16], det, invOut[16];
        int i;

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
                 m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
                 m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
                 m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
                  m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
                 m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
                 m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
                 m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
                  m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
                 m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
                 m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
                  m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
                  m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
                 m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
                 m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
                  m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
                  m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if (det == 0) {
            bCanReverse = false;
            return re;
        }

        det = 1.0 / det;

        for (i = 0; i < 16; i++)
            invOut[i] = inv[i] * det;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                re.data[i][j] = invOut[i * 4 + j];
            }
        }

        return re;
    }


    template <bool IS_SQUARE = N == M>
    __device__ __host__ std::enable_if_t<IS_SQUARE, Matrix> inversed() const {
        Matrix L = *this;
        Matrix R = Matrix::identity();
        for (int i = 0; i < N - 1; i++) {
            // make sure R[i][i] != 0
            int max_abs_index = i;
            DataT max_abs_val = R[{i, i}];
            if (max_abs_val < 0)
                max_abs_val = -max_abs_val;

            for (int j = i + 1; j < N; j++) {
                // 只要别带来数值精度问题，都没啥关系。DataT：1e-38
                if (max_abs_val > 1e-10)
                    break;
                DataT abs_val = R[{j, i}];
                if (abs_val < 0)
                    abs_val = -abs_val;
                if (abs_val > max_abs_val) {
                    max_abs_val = abs_val;
                    max_abs_index = j;
                }
            }

            if (max_abs_index != i) {
                std::swap(L.row(i), L.row(max_abs_index));
                std::swap(R.row(i), R.row(max_abs_index));
            }

            DataT head = L[{i, i}];
            for (int j = i + 1; j < N; j++) {
                DataT div = L[{j, i}] / head;
                // TODO(wangweihao):
                auto temp1 = (L.row(i) * div);
                auto temp2 = (R.row(i) * div);
                L.row(j) -= temp1.row(0);
                R.row(j) -= temp2.row(0);
            }
        }

        for (int i = N - 1; i > 0; i--) {
            DataT head = L[{i, i}];
            for (int j = i - 1; j >= 0; j--) {
                DataT div = L[{j, i}] / head;
                //     auto temp1 = (L.row(i) * div);
                auto temp2 = (R.row(i) * div);
                //     L.row(j) -= temp1.row(0);
                R.row(j) -= temp2.row(0);
            }
        }

        for (int i = 0; i < N; i++) {
            DataT div = L[{i, i}];
            //      L.row(i) /= div;
            R.row(i) /= div;
        }

        return R;
    }

    template <typename NewType>
    Matrix<M, N, NewType> map(const std::function<NewType(DataT)> &closure) const {
        Matrix<M, N, NewType> newmat;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                newmat[{i, j}] = closure((*this)[{i, j}]);
            }
        }
        return newmat;
    }

    template <typename NewType, typename = std::enable_if_t<std::is_convertible_v<DataT, NewType>>>
    Matrix<M, N, NewType> as() const {
        return map<NewType>([](auto val) { return NewType(val); });
    }

    template <typename O> __device__ __host__ explicit Matrix(const Matrix<M, N, O> &other) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                data[i][j] = float(other[{i, j}]);
            }
        }
    }

    template <bool IS_VEC4 = M == 4 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC4, Matrix<3, 1, DataT>> to_vec3_as_pos() const {
        Matrix<3, 1, DataT> mat;
        for (int i = 0; i < 3; i++)
            mat[i] = (*this)[i] / (*this)[3];
        return mat;
    }

    template <bool IS_VEC4 = M == 4 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC4, Matrix<3, 1, DataT>> to_vec3_as_dir() const {
        Matrix<3, 1, DataT> mat;
        for (int i = 0; i < 3; i++)
            mat[i] = (*this)[i];
        return mat;
    }

    template <bool IS_VEC3 = M == 3 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC3, Matrix<4, 1, DataT>> to_vec4_as_pos() const {
        Matrix<4, 1, DataT> mat;
        for (int i = 0; i < 3; i++)
            mat[i] = (*this)[i];
        mat[3] = DataT(1.0);
        return mat;
    }

    template <bool IS_VEC3 = M == 3 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC3, Matrix<4, 1, DataT>> to_vec4_as_dir() const {
        Matrix<4, 1, DataT> mat;
        for (int i = 0; i < 3; i++)
            mat[i] = (*this)[i];
        mat[3] = DataT(0);
        return mat;
    }

    template <bool IS_VEC3 = M == 3 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC3, Matrix<3, 1, DataT>>
    symmetric_vector(const Matrix<3, 1, DataT> &normalized_normal) {
        return ((data[0][0] * normalized_normal.data[0][0] + data[0][1] * normalized_normal.data[0][1] +
                 data[0][2] * normalized_normal.data[0][2]) *
                DataT(2)) *
                   normalized_normal -
               *this;
    }

    template <int SIZE, int Index, int... Indices>
    __device__ __host__ void make_slice(MatrixSlice<SIZE, 1, DataT> &slice) {
        slice.data[SIZE - 1 - sizeof...(Indices)][0] = (DataT *)&data[Index][0];
        make_slice<SIZE, Indices...>(slice);
    }

    template <int SIZE> __device__ __host__ void make_slice(MatrixSlice<SIZE, 1, DataT> &slice) {}

    template <int... Indices, bool IS_VECTOR = N == 1>
    __device__ __host__ std::enable_if_t<IS_VECTOR, MatrixSlice<sizeof...(Indices), 1, DataT>> slice() {

        MatrixSlice<sizeof...(Indices), 1, DataT> slice;
        make_slice<sizeof...(Indices), Indices...>(slice);
        return slice;
    }

    template <bool IS_VEC2_OR_GT = M >= 2 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC2_OR_GT, MatrixSlice<2, 1, DataT>> xy() {
        return slice<0, 1>();
    }

    template <bool IS_VEC3_OR_GT = M >= 3 && N == 1>
    __device__ __host__ std::enable_if_t<IS_VEC3_OR_GT, MatrixSlice<3, 1, DataT>> xyz() {
        return slice<0, 1, 2>();
    }

    __device__ __host__ MatrixSlice<1, N, DataT> row(int i) {
        MatrixSlice<1, N, DataT> out;
        for (int j = 0; j < N; j++) {
            out.data[0][j] = &data[i][j]; // TODO(wangweihao): ????
        }
        return out;
    }

    __device__ __host__ MatrixSlice<M, 1, DataT> col(int j) {
        MatrixSlice<M, 1, DataT> out;
        for (int i = 0; i < M; i++) {
            out.data[i][0] = &data[i][j];
        }
        return out;
    }

    __device__ __host__ Matrix set_col(int j, Matrix<N, 1, DataT> slice) {
        for (int i = 0; i < N; i++) {
            data[i][j] = slice.data[i][0];
        }
        return *this;
    }
};

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> MatrixSlice<M, N, DataT>::clone() const {
    Matrix<M, N, DataT> out;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            out.data[i][j] = *(this->data[i][j]);
    return out;
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> operator*(DataT k, const Matrix<M, N, DataT> &rhs) {
    return rhs * k;
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> operator+(DataT k, const Matrix<M, N, DataT> &rhs) {
    return rhs + k;
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> operator-(DataT k, const Matrix<M, N, DataT> &rhs) {
    return rhs - k;
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> operator-(const Matrix<M, N, DataT> &rhs) {
    return rhs * DataT(-1);
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> elemul(const Matrix<M, N, DataT> &lhs,
                                               const Matrix<M, N, DataT> &rhs) {
    Matrix<M, N, DataT> out;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[{i, j}] = lhs[{i, j}] * rhs[{i, j}];
        }
    }
    return out;
}
template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> elemax(const Matrix<M, N, DataT> &lhs,
                                               const Matrix<M, N, DataT> &rhs) {
    Matrix<M, N, DataT> out;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[{i, j}] = max(lhs[{i, j}], rhs[{i, j}]);
        }
    }
    return out;
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> elemin(const Matrix<M, N, DataT> &lhs,
                                               const Matrix<M, N, DataT> &rhs) {
    Matrix<M, N, DataT> out;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[{i, j}] = min(lhs[{i, j}], rhs[{i, j}]);
        }
    }
    return out;
}

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> elediv(const Matrix<M, N, DataT> &lhs,
                                               const Matrix<M, N, DataT> &rhs) {
    Matrix<M, N, DataT> out;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[{i, j}] = lhs[{i, j}] / rhs[{i, j}];
        }
    }
    return out;
}

template <int N, typename DataT> using Vector = Matrix<N, 1, DataT>;

// 如何只对某个成员函数进行特化？
template <typename DataT> using Vec2 = Vector<2, DataT>;
template <typename DataT> using Vec3 = Vector<3, DataT>;
template <typename DataT> using Vec4 = Vector<4, DataT>;
template <typename DataT> using Mat3 = Matrix<3, 3, DataT>;
template <typename DataT> using Mat4 = Matrix<4, 4, DataT>;
template <typename DataT> using RGBColor = Vec3<DataT>;
template <typename DataT> using RGBAColor = Vec4<DataT>;

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Mat3f = Mat3<float>;
using Mat4f = Mat4<float>;
using RGBColorF = Vec3f;
using RGBAColorF = Vec4f;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;
using Mat3d = Mat3<double>;
using Mat4d = Mat4<double>;
using RGBColorD = Vec3d;
using RGBAColorD = Vec4d;

template <typename DataT>
__device__ __host__ Vec3<DataT> cross_product(const Vec3<DataT> &vec31, const Vec3<DataT> &vec32) {
    Vec3<DataT> temp;
    temp[0] = vec31[1] * vec32[2] - vec32[1] * vec31[2];
    temp[1] = vec31[2] * vec32[0] - vec32[2] * vec31[0];
    temp[2] = vec31[0] * vec32[1] - vec32[0] * vec31[1];
    return temp;
}

template <int N, typename DataT>
__device__ __host__ DataT dot_product(const Vector<N, DataT> &vec1, const Vector<N, DataT> &vec2) {
    DataT sum{0};
    for (int i = 0; i < N; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
};

template <int M, int N, typename DataT>
__device__ __host__ Matrix<M, N, DataT> elepow(const Matrix<M, N, DataT> &src, DataT power) {
    Matrix<M, N, DataT> out(src);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out.data[i][j] = std::pow(out.data[i][j], power);
        }
    }
    return out;
}

template <typename DataT> Vec3<DataT> glm_to_matrix(const glm::vec3 &val) {
    return {DataT(val.x), DataT(val.y), DataT(val.z)};
}

template <typename DataT> __device__ __host__ DataT DifferenceOfProducts(DataT a, DataT b, DataT c, DataT d) {
    DataT cd = c * d;
    DataT err = std::fma(float(-c), float(d), float(cd));
    DataT dop = std::fma(float(a), float(b), float(-cd));
    return dop + err;
}

template <typename DataT>
__device__ __host__ Vec3<DataT> cross_product_difference(Vec3<DataT> vec31, Vec3<DataT> vec32) {
    Vec3<DataT> temp;
    temp[0] = DifferenceOfProducts(vec31[1], vec32[2], vec32[1], vec31[2]);
    temp[1] = DifferenceOfProducts(vec31[2], vec32[0], vec32[2], vec31[0]);
    temp[2] = DifferenceOfProducts(vec31[0], vec32[1], vec32[0], vec31[1]);
    return temp;
}

template <typename DataT>
__device__ __host__ DataT dot_product_difference(Vec3<DataT> vec31, Vec3<DataT> vec32) {
    DataT sum{0};
    sum = DifferenceOfProducts(DifferenceOfProducts(vec31[0], vec32[0], -vec31[1], vec32[1]), DataT(1),
                               -vec31[2], vec32[2]);
    return sum;
}

// referred to scipy's impl..
template <typename DataT> __host__ Mat4<DataT> quaternion_to_matrix(const Vec4<DataT> &quat) {
    auto x = quat[0];
    auto y = quat[1];
    auto z = quat[2];
    auto w = quat[3];

    auto x2 = x * x;
    auto y2 = y * y;
    auto z2 = z * z;
    auto w2 = w * w;

    auto xy = x * y;
    auto zw = z * w;
    auto xz = x * z;
    auto yw = y * w;
    auto yz = y * z;
    auto xw = x * w;

    return {{x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), 0},
            {2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw), 0},
            {2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2, 0},
            {0, 0, 0, 1}};
}

} // namespace rt
#endif
