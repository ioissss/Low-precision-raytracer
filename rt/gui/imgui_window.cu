#include <Windows.h>
#include <glad/glad.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include "imgui_window.hpp"
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "controller.hpp"
#include "rtrt/cuda.hpp"
#include "rtrt/loader.hpp"
#include "rtrt/memory.hpp"
#include "utils/gldebug.hpp"

#include <chrono>
#include <stdio.h>
#include <string>
#include <list>
#include <queue>

#include "stb_image.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

namespace rt {

DebugInfo debug_info;

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

std::string open_file(const char *filter) {
    char buffer[4096];

    OPENFILENAME dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.lpstrFile = buffer;
    dialog.lpstrFile[0] = '\0';
    dialog.nMaxFile = sizeof(buffer);
    dialog.lpstrFilter = filter;
    dialog.nFilterIndex = 1;
    dialog.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    GetOpenFileName(&dialog);

    return std::string(dialog.lpstrFile);
}

template <typename DataT> static Mat4<DataT> convert_matrix(glm::mat4 data) {
    Mat4f temp;
    memcpy(temp.data, &data, sizeof(data));
    return temp.as<DataT>();
}

struct VectorOrValue {
    static VectorOrValue Value(float val) {
        VectorOrValue f;
        f.value = val;
        f.is_vector = false;
        return f;
    }
    static VectorOrValue Vector() {
        VectorOrValue f;
        f.is_vector = true;
        return f;
    }
    bool is_vector;
    float value;
    std::vector<std::pair<std::string, VectorOrValue>> vector;
};

template <typename DataT, bool IS_F16 = std::is_same_v<DataT, float16>,
          bool IS_F32 = std::is_same_v<DataT, float>, typename = std::enable_if_t<IS_F16 || IS_F32>>
class Renderer {
    static constexpr int MAX_DIRECT_LIGHT = 4;

    int width, height;
    std::pair<std::string, VectorOrValue> statistic;

    static double timing(std::function<void(void)> action) {
        auto begin_time = std::chrono::high_resolution_clock::now();
        action();
        auto end_time = std::chrono::high_resolution_clock::now();
        return 1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
    }

  public:
    struct Settings {
        bool gi_on = false;
        bool traced_primary_ray = false;
        float svgf_color_mix_weight = 0.1;
        float svgf_moments_mix_weight = 0.1;
        float taa_mix_weight = 1;
    } dynamic_settings;

    DemoSetting demo_setting;

    UniformLocation uniform_location{get_gbuffer_shader()};

    std::unique_ptr<FrameBuffer> fb_draw_target;
    std::unique_ptr<CUDASurface> tex_draw_target;

    // 内部数据结构, 方便传参
    std::shared_ptr<CUDARenderGIData<DataT>> render_data;
    std::shared_ptr<RDResource<DataT>> rd_resource;
    std::shared_ptr<RTRTProcedureBuffer<DataT, MAX_DIRECT_LIGHT>> buffer;

    // 每次的场景
    std::shared_ptr<RDScene<DataT>> current_rdscene;

    RDCamera<DataT> last_rd_camera;

    std::pair<std::string, VectorOrValue> get_statistic() const { return statistic; }

    void render_cuda() {
        auto out_surf_obj = tex_draw_target->get_surface_object_wrapper();
        auto out = out_surf_obj.get();

        CUDARenderGISettings<DataT> settings{width, height, current_rdscene->camera};

        int n_total_size = settings.width * settings.height;
        int block_size = 32 * 2;
        int grid_size = (n_total_size + block_size - 1) / block_size;

        render_data->map();

        auto buffer_ref = buffer->get_ref();
        auto render_input = render_data->get_render_input(true);

        buffer->clear_intensity();
        buffer->clear_di_intensity();
        buffer->clear_shade_commands();
        buffer->clear_trace_gi_commands();
        buffer->clear_filterable_cache();

        float temp_time;
        temp_time = timing([&]() {
            generate_temporal_map_step1<DataT, MAX_DIRECT_LIGHT>
                <<<grid_size, block_size>>>(render_input, buffer_ref, settings, rand());
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())

            generate_temporal_map_step2<DataT, MAX_DIRECT_LIGHT>
                <<<grid_size, block_size>>>(render_input, buffer_ref, settings);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())
        });
        statistic.second.vector.push_back({"Generate Temporal Map", VectorOrValue::Value(temp_time)});

        temp_time = timing([&]() {
            if (dynamic_settings.gi_on) {
                shade<DataT, MAX_DIRECT_LIGHT, true, false>
                    <<<grid_size, block_size>>>(render_input, buffer_ref, 0, rand(), settings);
                RT_CHECK_CUDA(cudaDeviceSynchronize())
                RT_CHECK_CUDA(cudaGetLastError())
            } else {
                shade<DataT, MAX_DIRECT_LIGHT, true, true>
                    <<<grid_size, block_size>>>(render_input, buffer_ref, 0, rand(), settings);
                RT_CHECK_CUDA(cudaDeviceSynchronize())
                RT_CHECK_CUDA(cudaGetLastError())
            }
        });
        statistic.second.vector.push_back({"Pixel Shade #1", VectorOrValue::Value(temp_time)});

        temp_time = timing([&]() {
            trace_di_light<DataT, MAX_DIRECT_LIGHT>
                <<<grid_size * MAX_DIRECT_LIGHT, block_size>>>(render_input, buffer_ref, false, settings);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())
        });
        statistic.second.vector.push_back({"Trace DI Ray #1", VectorOrValue::Value(temp_time)});

        temp_time = timing([&]() {
            accumulate_di_light<DataT, MAX_DIRECT_LIGHT><<<grid_size, block_size>>>(buffer_ref, 0, settings);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())
        });
        statistic.second.vector.push_back({"Accumulate DI #1", VectorOrValue::Value(temp_time)});

        if (dynamic_settings.gi_on) {
            buffer->clear_di_intensity();

            temp_time = timing([&]() {
                trace_gi<DataT, MAX_DIRECT_LIGHT>
                    <<<grid_size, block_size>>>(render_input, buffer_ref, 0, settings);
                RT_CHECK_CUDA(cudaDeviceSynchronize())
                RT_CHECK_CUDA(cudaGetLastError())
            });
            statistic.second.vector.push_back({"Trace GI #1", VectorOrValue::Value(temp_time)});

            temp_time = timing([&]() {
                shade<DataT, MAX_DIRECT_LIGHT, false, true>
                    <<<grid_size, block_size>>>(render_input, buffer_ref, 1, rand(), settings);
                RT_CHECK_CUDA(cudaDeviceSynchronize())
                RT_CHECK_CUDA(cudaGetLastError())
            });
            statistic.second.vector.push_back({"Shade #2", VectorOrValue::Value(temp_time)});

            buffer->clear_shade_commands();
            buffer->clear_trace_gi_commands();

            temp_time = timing([&]() {
                trace_di_light<DataT, MAX_DIRECT_LIGHT>
                    <<<grid_size * MAX_DIRECT_LIGHT, block_size>>>(render_input, buffer_ref, true, settings);
                RT_CHECK_CUDA(cudaDeviceSynchronize())
                RT_CHECK_CUDA(cudaGetLastError())
            });
            statistic.second.vector.push_back({"Trace DI #2", VectorOrValue::Value(temp_time)});

            temp_time = timing([&]() {
                accumulate_di_light<DataT, MAX_DIRECT_LIGHT>
                    <<<grid_size, block_size>>>(buffer_ref, 1, settings);
                RT_CHECK_CUDA(cudaDeviceSynchronize())
                RT_CHECK_CUDA(cudaGetLastError())
            });
            statistic.second.vector.push_back({"Accumulate DI #2", VectorOrValue::Value(temp_time)});
        }

        temp_time = timing([&]() {
            write_clean_color<DataT, MAX_DIRECT_LIGHT>
                <<<grid_size, block_size>>>(render_input, buffer_ref, false, settings, demo_setting);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())

            if (demo_setting.svgf) {
                svgf_denoise<DataT, MAX_DIRECT_LIGHT>(
                    buffer_ref.mul_gi_colored, buffer_ref.gi_colored_svgf, render_input, buffer_ref, settings,
                    dynamic_settings.svgf_color_mix_weight, dynamic_settings.svgf_moments_mix_weight);

                svgf_denoise<DataT, MAX_DIRECT_LIGHT>(
                    buffer_ref.mul_gi_white, buffer_ref.gi_white_svgf, render_input, buffer_ref, settings,
                    dynamic_settings.svgf_color_mix_weight, dynamic_settings.svgf_moments_mix_weight);
            }

            add_denoised_color<DataT, MAX_DIRECT_LIGHT>
                <<<grid_size, block_size>>>(out, render_input, buffer_ref, false, settings, demo_setting);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())
        });

        statistic.second.vector.push_back({"SVGF", VectorOrValue::Value(temp_time)});

        temp_time = timing([&]() {
            temporal_anti_aliasing<DataT><<<grid_size, block_size>>>(
                buffer_ref.color_inprogress, buffer_ref.taa_history_color, buffer_ref.taa_temporal_map, settings,
                dynamic_settings.taa_mix_weight);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())

            write_to_surface2d<DataT><<<grid_size, block_size>>>(out, buffer_ref.color_inprogress, settings);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())

            RT_CHECK_CUDA(cudaMemcpy(buffer_ref.taa_history_color, buffer_ref.color_inprogress,
                                     sizeof(*buffer_ref.taa_history_color) * buffer->taa_history_color.size(),
                                     cudaMemcpyDeviceToDevice))
        });
        statistic.second.vector.push_back({"TAA", VectorOrValue::Value(temp_time)});

        temp_time = timing([&]() {
            copy_last_frame_pixel_id<DataT, MAX_DIRECT_LIGHT>
                <<<grid_size, block_size>>>(out, render_input, buffer_ref, settings);
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())

            copy_last_frame_transform<DataT, MAX_DIRECT_LIGHT>
                <<<(render_data->objects_constant.size() + block_size - 1) / block_size, block_size>>>(
                    render_input, buffer_ref, render_data->objects_constant.size());
            RT_CHECK_CUDA(cudaDeviceSynchronize())
            RT_CHECK_CUDA(cudaGetLastError())
            buffer->last_frame_transform_W2C = render_data->transform_W2C;
        });
        // OpenGL 有 texture 数量限制，只能拷贝了
        statistic.second.vector.push_back({"Backup Temporal Data", VectorOrValue::Value(temp_time)});

        render_data->unmap();
    }

    void render_gbuffer() {
        const auto &camera = current_rdscene->camera;
        Mat4f transform_f32(camera.transform);
        glm::mat4 world_to_view =
            glm::transpose(glm::make_mat4(reinterpret_cast<const float *>(transform_f32.data)));
        glm::mat4 view_to_clip =
            glm::perspectiveFov<float>(camera.field_of_view_y, width, height, camera.z_near, camera.z_far);

        glm::mat4 world_to_clip = view_to_clip * world_to_view;
        render_data->transform_W2C = convert_matrix<DataT>(glm::transpose(world_to_clip));

        float time_of_gbuffer_gen = timing([&]() {
            if (dynamic_settings.traced_primary_ray) {
                auto trans = convert_matrix<DataT>(glm::inverse(world_to_clip));
                CUDARenderGISettings<DataT> settings{width, height, current_rdscene->camera};
                render_cuda_gbuffer(render_data, settings, trans);
            } else {

                glEnable(GL_DEPTH_TEST);
                glDisable(GL_DITHER);

                glUseProgram(get_gbuffer_shader());

                render_data->gbuffer->bind();

                glClearColor(0, 0, 0, 0);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glViewport(0, 0, width, height);

                glUniformMatrix4fv(uniform_location.world_to_clip_loc, 1, GL_FALSE,
                                   glm::value_ptr(world_to_clip));
                for (int i = 0; i < current_rdscene->objects_constants.size(); i++) {
                    auto objectid = current_rdscene->objects_constants[i].objectid;
                    current_rdscene->objects_constants[i].bind(uniform_location, i);
                    rd_resource->objects_vao[objectid]->draw();
                }

                render_data->gbuffer->unbind();
            }
        });

        statistic.first = "Rendering Statistic";
        statistic.second.is_vector = true;
        statistic.second.vector.push_back({"Primary Ray", VectorOrValue::Value(time_of_gbuffer_gen)});
    }

  public:
    void set_rdresource(std::shared_ptr<RDResource<DataT>> val) {
        this->rd_resource = val;
        this->render_data->rd_resource = val;
        this->render_data->fill_objects_bvh();
        this->render_data->fill_objects_cuda_vao();
    }

    void render(std::shared_ptr<RDScene<DataT>> rd_scene) {
        this->statistic.second.vector.clear();

        this->current_rdscene = rd_scene;
        this->render_data->rd_scene = rd_scene;

        this->render_gbuffer();
        this->render_cuda();
    }

    Renderer(int width, int height, int number_of_vaos) : width(width), height(height) {
        fb_draw_target = std::make_unique<FrameBuffer>();
        tex_draw_target = std::make_unique<CUDASurface>(width, height, fb_draw_target->index, GL_RGBA32F,
                                                        GL_COLOR_ATTACHMENT0, true);
        render_data = std::make_shared<CUDARenderGIData<DataT>>();
        render_data->gbuffer = std::unique_ptr<GBuffer>(new GBuffer(width, height, IS_F16));

        buffer = std::make_shared<RTRTProcedureBuffer<DataT, 4>>(width, height, 2, number_of_vaos);
    }

    ~Renderer() {
        tex_draw_target = nullptr;
        fb_draw_target = nullptr;
    }

    GLuint get_gbuffer_normal_texture() { return this->render_data->gbuffer->get_gl_normal_depth(); }

    GLuint get_render_target_texture() { return this->tex_draw_target->get_buffer_id(); }
};

class AbstractSceneExplorer {
  public:
    virtual ~AbstractSceneExplorer() = default;
    virtual void render() = 0;
};

template <typename DataT> class SceneExplorer : public AbstractSceneExplorer {
    MoveController x_controller{ImGuiKey_D, ImGuiKey_A};
    MoveController y_controller{ImGuiKey_E, ImGuiKey_C};
    MoveController z_controller{ImGuiKey_S, ImGuiKey_W};

    MoveController roll_controller{ImGuiKey_LeftArrow, ImGuiKey_RightArrow};

    HoldRotateController mouse_controller;

    std::unique_ptr<hierarchy::Scene<DataT>> scene;
    std::shared_ptr<hierarchy::Camera<DataT>> scene_camera;
    std::shared_ptr<hierarchy::Object<DataT>> extra_sun_light_parent;
    std::shared_ptr<hierarchy::Camera<DataT>> free_camera;
    std::shared_ptr<hierarchy::Object<DataT>> model_root;

    std::shared_ptr<hierarchy::Light<DataT>> extra_sun_light;

    std::shared_ptr<Renderer<DataT>> renderer;
    std::shared_ptr<RDResource<DataT>> rd_resource;
    std::shared_ptr<RDScene<DataT>> rd_scene;

    float skybox_delta_xy[2] = {0, 0};
    float skybox_exposure = 1.0;

    enum class CameraState { FREE, SCENE } camera_state = CameraState::SCENE;

    float model_orientation[3] = {0, 0, 0};

    double last_frame_duration = -1;
    std::chrono::steady_clock::time_point frame_begin_time;
    std::chrono::steady_clock::time_point model_loaded_time;

    float font_scale;
    int width;
    int height;

    std::string path;

    enum class LoadingState { BEFORE_LOADING, LOADING, LOADED } loading_state = LoadingState::BEFORE_LOADING;

    bool do_render = true;

    bool sun_light = false;
    float sun_light_direction[3] = {36.0 / 180 * M_PI, 36.0 / 180 * M_PI, 36.0 / 180 * M_PI};
    float sun_light_color[3] = {1, 1, 1};

    bool show_skybox_chk = false;
    std::string skybox_image_path = "";
    std::shared_ptr<RDTexture> skybox_texture;

  public:
    SceneExplorer(std::string path, int width, int height, float font_scale = 1)
        : font_scale(font_scale), width(width), height(height), path(path) {

        roll_controller.initial_speed = 0.1;
        roll_controller.acceleration = 0.2;
        roll_controller.max_speed = 0.6;
        load_model(path);
    }

    ~SceneExplorer() override = default;

    void render() {
        if (!do_render)
            return;
        if (last_frame_duration < 0) {
            frame_begin_time = std::chrono::high_resolution_clock::now();
        }

        handle_events();

        // 更新动画
        auto elapsed_time = 1e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now() - model_loaded_time)
                                       .count();
        scene->root_object->apply_animation(elapsed_time);

        // 构建场景
        rd_scene = scene->build_rendering_scene();

        if (show_skybox_chk && skybox_texture != nullptr) {
            rd_scene->skybox.delta_x = skybox_delta_xy[0];
            rd_scene->skybox.delta_y = skybox_delta_xy[1];
            rd_scene->skybox.texture = skybox_texture;
            rd_scene->skybox.exposure = DataT(skybox_exposure);
        }

        // 渲染
        renderer->render(rd_scene);

        // UI
        render_mainwindow();
        render_inspect_window();
        render_statistic_window();

        const auto end_time = std::chrono::high_resolution_clock::now();
        last_frame_duration =
            1e-6 * std::chrono::duration_cast<std::chrono::microseconds>(end_time - frame_begin_time).count();
        frame_begin_time = end_time;
    }

  protected:
    bool show_normal = false;

    void render_statistic_tree(const std::pair<std::string, VectorOrValue> &val) {
        if (val.second.is_vector) {
            if (ImGui::TreeNode(val.first.c_str())) {
                for (auto child : val.second.vector) {
                    render_statistic_tree(child);
                }
                ImGui::TreePop();
            }
        } else {
            ImGui::LabelText(val.first.c_str(), "%3.3f ms", val.second.value * 1000);
        }
    }

    void render_statistic_window() {
        ImGui::Begin("Statistic");
        ImGui::SetWindowFontScale(font_scale);
        ImGui::Text("%.0f FPS\n", 1 / last_frame_duration);

        render_statistic_tree(renderer->get_statistic());

        ImGui::End();
    }

    void render_inspect_window() {
        bool b_free_camera = camera_state == CameraState::FREE;

        ImGui::Begin("Inspect");
        ImGui::SetWindowFontScale(font_scale);
        ImGui::Checkbox("Show Normal", &show_normal);
        ImGui::Checkbox("Traced Primary Ray", &renderer->dynamic_settings.traced_primary_ray);

        if (show_normal) {
            render_texture("Normal (World Coordinate)", renderer->get_gbuffer_normal_texture());
        }

        ImGui::SliderAngle("Model Yaw", &model_orientation[0]);
        ImGui::SliderAngle("Model Pitch", &model_orientation[1]);
        ImGui::SliderAngle("Model Roll", &model_orientation[2]);

        ImGui::SliderFloat("TAA Weight", &renderer->dynamic_settings.taa_mix_weight, 0, 1);

        ImGui::Separator();
        ImGui::Checkbox("SVGF Denoiser", &renderer->demo_setting.svgf);
        if (renderer->demo_setting.svgf) {
            ImGui::SliderFloat("SVGF - Color", &renderer->dynamic_settings.svgf_color_mix_weight, 0, 1,
                               "%.3f");
            ImGui::SliderFloat("SVGF - Moments", &renderer->dynamic_settings.svgf_moments_mix_weight, 0, 1,
                               "%.3f");
        }

        ImGui::Separator();
        ImGui::Checkbox("Global Illumination", &renderer->dynamic_settings.gi_on);
        if (renderer->dynamic_settings.gi_on) {
            ImGui::Checkbox("Show Direct Out", &renderer->demo_setting.add_direct_out);
            ImGui::Checkbox("Show GI Colored", &renderer->demo_setting.add_gi_colored);
            ImGui::Checkbox("Show GI White", &renderer->demo_setting.add_gi_white);
            ImGui::Checkbox("Demodulate", &renderer->demo_setting.demodulate);
        }

        ImGui::Separator();
        ImGui::Checkbox("Sun Light", &sun_light);
        if (sun_light) {
            ImGui::SliderFloat3("Color RGB", sun_light_color, 0, 100, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderAngle("Direction #1", &sun_light_direction[0]);
            ImGui::SliderAngle("Direction #2", &sun_light_direction[1]);
            ImGui::SliderAngle("Direction #3", &sun_light_direction[2]);
        }
        ImGui::Checkbox("Skybox", &show_skybox_chk);
        if (show_skybox_chk) {
            if (ImGui::Button("Open (Spherical Map)")) {
                std::string new_path = open_file("Spherical Map (*.hdr)\0*.hdr\0");
                if (!new_path.empty())
                    skybox_image_path = new_path;

                int width;
                int height;
                int comp;
                uint8_t *data;
                data = stbi_load(skybox_image_path.c_str(), &width, &height, &comp, STBI_rgb_alpha);
                skybox_texture = std::make_shared<RDTexture>(data, width, height, true);
                stbi_image_free(data);
            }
            ImGui::SameLine();
            ImGui::Text("%s", skybox_image_path.c_str());
            ImGui::SliderFloat2("Offset", skybox_delta_xy, 0, 1, "%.3f");
            ImGui::SliderFloat("Skybox Exposure", &skybox_exposure, 0.01, 100, "%.4f",
                               ImGuiSliderFlags_Logarithmic);
        }

        ImGui::Separator();
        ImGui::Checkbox("Free Camera", &b_free_camera);
        if (b_free_camera) {
            camera_state = CameraState::FREE;
        } else {
            camera_state = CameraState::SCENE;
        }
        if (camera_state == CameraState::FREE) {
            auto [rx, ry, rz] = mouse_controller.get_xyz();
            float roll = roll_controller.get_value();
            float yaw_pitch_roll[3] = {ry, rx, roll};
            auto free_camera_pos = free_camera->parent.lock();
            ImGui::InputFloat3("Camera XYZ",
                               reinterpret_cast<float *>(glm::value_ptr(free_camera_pos->translation)));
            ImGui::InputFloat3("Yaw Pitch Roll", yaw_pitch_roll);
            ImGui::InputDouble("Field Of View", &mouse_controller.acc_z);
            mouse_controller.acc_x = yaw_pitch_roll[1];
            mouse_controller.acc_y = yaw_pitch_roll[0];
            roll_controller.accumulated_pos = yaw_pitch_roll[2];
        }

        ImGui::End();
    }

    void load_model(const std::string &path) {
        rd_resource = std::make_shared<RDResource<DataT>>();
        model_root = load_gltf2(path, *rd_resource);

        renderer = std::make_shared<Renderer<DataT>>(width, height, rd_resource->objects_vao.size());
        renderer->set_rdresource(rd_resource);

        scene = std::make_unique<hierarchy::Scene<DataT>>();
        scene->root_object = std::make_shared<hierarchy::Object<DataT>>();

        extra_sun_light_parent = std::make_shared<hierarchy::Object<DataT>>();
        extra_sun_light_parent->parent = scene->root_object;
        scene->root_object->children.push_back(extra_sun_light_parent);

        extra_sun_light = std::make_shared<hierarchy::Light<DataT>>();
        extra_sun_light->data.type = RDLight<DataT>::LightType::DIRECTIONAL;
        // 他的 parent 会在 handle events 里面设置

        // 最后一个加 model_root, 保证额外添加的光源优先展示
        scene->root_object->children.push_back(model_root);
        model_root->parent = scene->root_object;

        scene_camera = scene->search_camera();
        scene->active_camera = scene_camera;

        // Free Camera 要在搜索 Scene Camera 之后添加
        free_camera = std::make_shared<hierarchy::Camera<DataT>>();
        free_camera->parent = scene->root_object;
        scene->root_object->children.push_back(free_camera);

        auto free_camera_parent = free_camera->insert_parent(free_camera);
        free_camera_parent->translation.z = 2;
        free_camera_parent->quat_rotation = glm::angleAxis(float(M_PI / 2), glm::vec3(1, 0, 0));

        model_loaded_time = std::chrono::high_resolution_clock::now();
    }

    void render_texture(char const *title, GLuint textureid) {
        ImGui::Begin(title);
        ImGui::SetWindowFontScale(font_scale);
        ImTextureID texture = (void *)(size_t)textureid;
        auto cursor = ImGui::GetCursorScreenPos();
        ImGui::Image(texture, ImVec2{(float)width, (float)height}, ImVec2{0.f, 1.f}, ImVec2{1.f, 0.f});
        auto mouse = ImGui::GetMousePos();
        ImGui::Text("%.0f %.0f", mouse.x - cursor.x, height - 1 - (mouse.y - cursor.y));
        ImGui::End();
    }

    void render_mainwindow() { render_texture("Viewer", renderer->get_render_target_texture()); }

    void handle_events() {
        if (scene_camera == nullptr) {
            camera_state = CameraState::FREE;
        }

        if (camera_state == CameraState::FREE) {
            scene->active_camera = free_camera;
        } else {
            scene->active_camera = scene_camera;
        }

        if (camera_state == CameraState::FREE) {
            x_controller.receive_event(last_frame_duration);
            y_controller.receive_event(last_frame_duration);
            z_controller.receive_event(last_frame_duration);
            roll_controller.receive_event(last_frame_duration);
            mouse_controller.receive_event(last_frame_duration);

            auto [rx, ry, rz] = mouse_controller.get_xyz();
            free_camera->quat_rotation = glm::quat(glm::vec3(ry, rx, roll_controller.get_value()));
            free_camera->data.field_of_view_y = rz;

            auto free_camera_pos = free_camera->parent.lock();
            auto trans = free_camera_pos->transform_matrix();

            trans = glm::translate(trans, free_camera->quat_rotation * glm::vec3{x_controller.pop_value(),
                                                                                 y_controller.pop_value(),
                                                                                 z_controller.pop_value()});
            free_camera_pos->set_transform_matrix(trans);
        }

        if (sun_light) {
            extra_sun_light_parent->children = {extra_sun_light};
            extra_sun_light->parent = extra_sun_light_parent;
            for (int i = 0; i < 3; i++) {
                extra_sun_light->data.intensity[i] = DataT(sun_light_color[i]);
            }

            extra_sun_light->quat_rotation = glm::quat(glm::make_vec3(sun_light_direction));
        } else {
            extra_sun_light_parent->children.clear();
        }

        model_root->quat_rotation = glm::quat(glm::make_vec3(model_orientation));
    }
};

class ImGuiRTWindow {
    std::chrono::time_point<std::chrono::steady_clock> begin_time;
    std::chrono::time_point<std::chrono::steady_clock> model_loaded_time;

    ImGuiRTWindow() = default;

    float width_height[2] = {1024, 768};

    bool half_float = false;

    std::unique_ptr<AbstractSceneExplorer> explorer;

  public:
    void clean() { explorer = nullptr; }

    float font_scale = 1;

    static ImGuiRTWindow &get_instance() {
        static ImGuiRTWindow window;
        return window;
    }

    void operator=(const ImGuiRTWindow &) = delete;
    ImGuiRTWindow(const ImGuiRTWindow &) = delete;

    std::list<std::pair<ImVec4, std::string>> messages;

    void render_error_message() {
        ImGui::Begin("Messages");
        ImGui::SetWindowFontScale(font_scale);

        for (auto [color, msg]: messages) {
            ImGui::TextColored(color, "%s", msg.c_str());
        }
        ImGui::End();

        while (messages.size() > 100) {
            messages.pop_back();
        }

    }

    void render() {
        render_error_message();

        static bool first = true;
        ImGui::Begin("Static Settings");
        ImGui::SetWindowFontScale(font_scale);
        ImGui::Checkbox("Half Float", &half_float);
        ImGui::InputFloat2("Width x Height", width_height, "%.0f");

        try {
            if (ImGui::Button("Open")) {
                auto path = open_file("glTF 2.0 Model(*.glb, *.gltf)\0*.glb;*.gltf\0");
                if (!path.empty()) {
                    explorer = nullptr;

                    if (half_float) {
                        explorer = std::unique_ptr<AbstractSceneExplorer>(new SceneExplorer<float16>(
                            path, lround(width_height[0]), lround(width_height[1]), font_scale));
                    } else {
                        explorer = std::unique_ptr<AbstractSceneExplorer>(new SceneExplorer<float>(
                            path, lround(width_height[0]), lround(width_height[1]), font_scale));
                    }
                }
            }
        } catch (const std::exception& err) {
            messages.push_front({ImVec4(1, 1, 0, 1), err.what()});
        } catch (...) {
            messages.push_front({ImVec4(1, 1, 0, 0), "unknown error"});
        }
        ImGui::End();

        try {
            if (explorer) {
                explorer->render();
            }
        } catch (const std::exception &err) {
            messages.push_front({ImVec4(1, 1, 0, 1), err.what()});
        } catch (...) {
            messages.push_front({ImVec4(1, 1, 0, 0), "unknown error"});
        }
        
    }
};

int run_imgui() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    const char *glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#endif
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow *window = glfwCreateWindow(1280, 720, "F16 RTRT Explorer", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
#ifndef NDEBUG
    enable_gl_debug();
#endif

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    float xscale, yscale;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    float dpi = std::min(xscale, yscale);

    ImGui::GetStyle().ScaleAllSizes(dpi);
    ImGuiRTWindow::get_instance().font_scale = dpi;

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiRTWindow::get_instance().render();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGuiRTWindow::get_instance().clean();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

} // namespace rt