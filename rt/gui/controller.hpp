#pragma once

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include "math/matrix.hpp"

#include <vector>

namespace rt {

class MoveController {
  public:
    // s^(-1)
    double initial_speed = 0.02;

    // s^(-2)
    double acceleration = 4;

    // max speed (s^-1)
    double max_speed = 30;

    double max_position = std::numeric_limits<double>::infinity();
    double min_position = -std::numeric_limits<double>::infinity();

  protected:
    ImGuiKey key_plus;
    ImGuiKey key_minus;
    int last_state = 0;
    double current_speed = 0;

  public:
    double accumulated_pos = 0;

    MoveController(ImGuiKey keyPlus, ImGuiKey keyMinus) : key_plus(keyPlus), key_minus(keyMinus) {}

    void clear() {
        current_speed = 0;
        accumulated_pos = 0;
        last_state = 0;
    }

    double get_value() const { return accumulated_pos; }

    double pop_value() {
        auto backup = accumulated_pos;
        accumulated_pos = 0;
        return backup;
    }

    void receive_event(double last_frame_duration) {
        int current_state = 0;

        if (ImGui::IsKeyDown(key_plus)) {
            current_state = 1;
        } else if (ImGui::IsKeyDown(key_minus)) {
            current_state = -1;
        }

        if (current_state != last_state || current_state == 0) {
            current_speed = current_state * initial_speed;
        }

        last_state = current_state;

        double delta_pos = 0;
        if (current_speed == max_speed * current_state) {
            delta_pos = current_speed * last_frame_duration;
        } else if (abs(current_speed + current_state * last_frame_duration * acceleration) > max_speed) {
            // 如果这次加速将超过最高速度
            double tIntermediate = (max_speed - abs(current_speed)) / (last_frame_duration * acceleration);
            delta_pos += (2 * current_speed + current_state * tIntermediate * acceleration) * tIntermediate / 2;
            delta_pos += (last_frame_duration - tIntermediate) * max_speed * current_state;
            current_speed = max_speed * current_state;
        } else {
            delta_pos =
                (2 * current_speed + current_state * last_frame_duration * acceleration) * last_frame_duration / 2;
            current_speed += current_state * last_frame_duration * acceleration;
            accumulated_pos += delta_pos;
        }

        accumulated_pos += delta_pos;
        if (accumulated_pos > max_position)
            accumulated_pos = max_position;
        if (accumulated_pos < min_position)
            accumulated_pos = min_position;
    }
};

// 经典 WASD - 鼠标 的控制方式
 class HoldRotateController {
    int last_x_pos = 0;
    int last_y_pos = 0;
    double last_wheel = 0;

    void clip() {
        // 不至于一帧转好几圈吧
        // 并且，即使出现了这种情况，问题也不大。。
        if (acc_x < M_PI) {
            acc_x += 2 * M_PI;
        }
        if (acc_x > M_PI) {
            acc_x -= 2 * M_PI;
        }
        if (acc_y > y_max) {
            acc_y = y_max;
        }
        if (acc_y < y_min) {
            acc_y = y_min;
        }

        if (acc_z < z_min) {
            acc_z = z_min;
        }

        if (acc_z > z_max) {
            acc_z = z_max;
        }
    }

  public:
    double acc_x = 0;
    double acc_y = 0;
    double acc_z = M_PI * 0.3;

    double x_sensitivity = 0.001f;
    double y_sensitivity = 0.001f;
    double z_sensitivity = 0.02f;

    double y_min = -double(0.9 * M_PI / 2);
    double y_max = +double(0.9 * M_PI / 2);

    double z_min = M_PI * 0.2;
    double z_max = M_PI * 0.7;

    bool is_down = false;

    std::tuple<double, double, double> get_xyz() { return {acc_x, acc_y, acc_z}; }

    void clear() {
        acc_x = acc_y = 0;
        acc_z = M_PI * 0.3;
    }

    void receive_event(double last_frame_duration)
    {
        auto pos = ImGui::GetMousePos();
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            if (!is_down) {
                this->last_x_pos = pos.x;
                this->last_y_pos = pos.y;
                is_down = true;
            }
        }

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            is_down = false;
        }



        if (is_down) {
            acc_x += (pos.x - last_x_pos) * x_sensitivity;
            acc_y += (pos.y - last_y_pos) * y_sensitivity;

            this->last_x_pos = pos.x;
            this->last_y_pos = pos.y;
            this->last_wheel = ImGui::GetIO().MouseWheel;
        }

        
        acc_z += this->z_sensitivity * ImGui::GetIO().MouseWheel;
        this->clip();
    }
};
} // namespace rt