#include "math/matrix.hpp"
#include "math/number.hpp"

namespace rt {
constexpr float eps_shader = 1e-5;

// V: view
// H: half
// L: light
// N: normal
// normalized vectors
template <typename DataT>
__device__ DataT specularBRDF(Vec3<DataT> V, Vec3<DataT> H, Vec3<DataT> L, Vec3<DataT> N, DataT alpha) {
    DataT HdotL = dot_product(H, L);
    DataT HdotV = dot_product(H, V);
    DataT NdotH = dot_product(N, H);

    if (NdotH <= DataT(0) || HdotL <= DataT(0) || HdotV <= DataT(0)) {
        return 0;
    } else {

        DataT absNdotL = abs(dot_product(N, L));
        DataT absNdotV = abs(dot_product(N, V));
        DataT a2 = alpha * alpha;

        DataT div1 = (absNdotL + sqrt(max<DataT>(0, (a2 + (DataT(1) - a2) * absNdotL * absNdotL))));
        DataT div2 = (absNdotV + sqrt(max<DataT>(0, (a2 + (DataT(1) - a2) * absNdotV * absNdotV))));

        auto D_val = a2 / (DataT(M_PI) * pow(NdotH * NdotH * (a2 - DataT(1)) + DataT(1), DataT(2)));

        return D_val / div1 / div2;
    }
}

template <typename DataT> struct BRDF {
    DataT colored;
    DataT white;

    __device__ BRDF() = default;
    template <typename T>
    __device__ BRDF(const BRDF<T> &b) : colored(DataT(b.colored)), white(DataT(b.white)) {}
    __device__ BRDF(DataT colored, DataT white): colored(colored), white(white) {}

    // ���ڻ������������� I = {i0, i1, ...}
    // \sum i in I { base_color * intensity_i * cosine_i * `colored`_i / pdf_i +  
    //               white * intensity_i * cosine_i * `white`_i / pdf_i }
    // Ҳ����
    // base_color * \sum i in I { intensity_i * cosine_i * `colored`_i / pdf_i } +
    // white * \sum i in I { intensity * i * cosine_i * `white`_i / pdf_i }


    __device__ RGBColor<DataT> get_brdf(const RGBColor<DataT> &base_color) {
        return base_color * colored + RGBColor<DataT>(1, 1, 1) * white;
    }
};

template <typename DataT>
__device__ BRDF<DataT> material_brdf(
    // material
    DataT metallic, DataT roughness,

    // normalized view direction
    Vec3<DataT> V,

    // normalized light direction
    Vec3<DataT> L,

    // normalized surface normal
    Vec3<DataT> N) {
    // �������� glTF 2.0 Specification �жԲ��ʵĹ涨д��
    // ��Ϊ�� normal map �Ĺ�ϵ�����ﲻ�ܼ� dot_product(V, H) < N
    if (dot_product(L, N) < DataT(0)) {
        return {DataT(0), DataT(0)};
    } else {
        Vec3<DataT> H = (L + V).normalized();
        DataT VdotH = dot_product(V, H);
        DataT pow5_of_1_sub_vh = pow(DataT(max(DataT(0), DataT(1) - abs(VdotH))), DataT(5));
        DataT alpha = roughness * roughness;

        DataT layer = specularBRDF(V, H, L, N, alpha); // ���߷ֲ� * Shadow-Masking / ��һ��

        // ��ģ����һ�ֻ�ϲ��ʣ����Կ�����һ�㱡�������� + ľ��
        // ����ķ����ǰ�ɫ�ģ��ǽ����ķ��������ⶼӦ���ǰ�ɫ�ģ���IOR �� 1.5���� roughness
        // ����������֮��Ĺ���Ͷ�䵽ľ���ϣ����о��ȵ�������
        DataT dielectric_f0 = 0.04;
        DataT dielectric_fr = dielectric_f0 + DataT(1 - dielectric_f0) * pow5_of_1_sub_vh;
        DataT dielectric_white = dielectric_fr * layer;
        DataT dielectric_colored = (DataT(1) - dielectric_fr) * DataT(1 / M_PI);

        // �������ʡ������� f0 ������ɫ�ģ������������Ȼ��ȫ����
        DataT metal_white = layer * pow5_of_1_sub_vh;
        DataT metal_colored = layer * (DataT(1) - pow5_of_1_sub_vh);

        return {metal_colored * metallic + dielectric_colored * (DataT(1) - metallic),
                metal_white * metallic + dielectric_white * (DataT(1) - metallic)};
    }
}

// ��Ϊ����������Ӧ�ǽ������߷ǽ����е�һ�������Կ�����һЩ�򵥵ķ������������������߷ǽ���
// ���� random() < metallic -> metallic, otherwise -> dielectric, random ~ U(0, 1)
// ����ǽ�����ֱ�����վ��淴�䴦��  in * (`(DataT(1) - pow5_of_1_sub_vh)` * base_color + `pow5_of_1_sub_vh` * white)
// ����Ƿǽ���������Ҫ���в������Ը��� p ����Ϊ�������࣬1-p ����Ϊ���淴����
// ������Ϊ�������࣬��ֱ�ӵ��� �� �ĺ�������
// ������Ϊ���淴���࣬����øú��� in * `dielectric_fr` * white����Ϊpdf�����庯����������diffuse���ֲ��ÿ�����
// �����Ϊ���Դ������
template <typename DataT>
__device__ BRDF<DataT> glassy_brdf(
    // material
    DataT metallic,

    // normalized view direction
    Vec3<DataT> V,

    // normalized light direction
    Vec3<DataT> L,

    // normalized surface normal
    Vec3<DataT> N) {
    // �������� glTF 2.0 Specification �жԲ��ʵĹ涨д��
    // ��Ϊ�� normal map �Ĺ�ϵ�����ﲻ�ܼ� dot_product(V, H) < N
    if (dot_product(L, N) < DataT(0)) {
        return {DataT(0), DataT(0)};
    } else {
        Vec3<DataT> H = (L + V).normalized();
        DataT VdotH = dot_product(V, H);
        DataT pow5_of_1_sub_vh = pow(DataT(max(DataT(0), DataT(1) - abs(VdotH))), DataT(5));
        
        DataT dielectric_f0 = 0.04;
        DataT dielectric_fr = dielectric_f0 + DataT(1 - dielectric_f0) * pow5_of_1_sub_vh;
        DataT dielectric_white = dielectric_fr;
        DataT dielectric_colored = 0;
       
        DataT metal_white = pow5_of_1_sub_vh;
        DataT metal_colored = (DataT(1) - pow5_of_1_sub_vh);

        return {metal_colored * metallic + dielectric_colored * (DataT(1) - metallic),
                metal_white * metallic + dielectric_white * (DataT(1) - metallic)};
    }
}



} // namespace rt