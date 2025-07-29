#pragma once

#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <filesystem>
#include "FilesystemManager.h"

class LoggerSystem {
public:
    explicit LoggerSystem(const FilesystemManager& fsm)
        : fsm_(fsm), log_file_(fsm.get_run_info_file()), indent_(0)
    {
        std::filesystem::create_directories(std::filesystem::path(log_file_).parent_path());
        log_stream_.open(log_file_);
        start_time_ = std::chrono::steady_clock::now();
    }

    ~LoggerSystem() {
        if (log_stream_.is_open()) log_stream_.close();
    }

    // ──────────────── 样式功能 ─────────────────
    void log_boxed_title(const std::string& title) {
        std::string line = "══════════════════════════════════════════════════════════════════════════";
        log_raw("╔" + line + "╗");
        log_raw("║" + center_text(title, 74) + "║");
        log_raw("╚" + line + "╝");
    }

    void log_section_title(const std::string& section) {
        log_raw("");
        log_boxed_title(section);
    }

    void print_header(const std::string& title) {
        log_raw("");
        log_boxed_title(title);
    }

    void print_config(int order, int meshN, int cpus) {
        log_kv("Order", order);
        log_kv("Mesh N", meshN);
        log_kv("CPU Cores", cpus);
        log_kv("Solution File", fsm_.get_solution_file(0, meshN));
        log_kv("Error Log", fsm_.get_error_log_file());
        log_kv("Config File", fsm_.get_config_file());
        log_kv("Run Info", fsm_.get_run_info_file());
    }

    // ──────────────── 计时功能 ─────────────────

    void start_stage(const std::string& stage_name) {
        stage_name_ = stage_name;
        stage_start_ = std::chrono::steady_clock::now();
        log_timed("Start " + stage_name);
    }

    void end_stage() {
        auto elapsed = elapsed_ms(stage_start_);
        log_timed("Finished " + stage_name_ + "  (elapsed: " + std::to_string(elapsed) + " ms)");
    }

    void log_time_step(int step, double t, double dt) {
        step_start_ = std::chrono::steady_clock::now();
        std::stringstream ss;
        ss << "Time Step " << step
        << " │ t = " << std::fixed << std::setprecision(6) << t
        << " sec │ Δt = " << std::setprecision(6) << dt
        << " sec";
        log_boxed_line(ss.str());
    }

    void end_time_step() {
        auto elapsed = elapsed_ms(step_start_);
        log_append_to_last_box(" Elapsed: " + format_duration(elapsed));
    }

    void log_picard_iteration(int iter) {
        picard_start_ = std::chrono::steady_clock::now();
        log_raw(indent_string() + "── Picard Iteration " + std::to_string(iter) + " ────────────────────────────────────────");
        increase_indent();
    }

    void end_picard_iteration() {
        auto elapsed = elapsed_ms(picard_start_);
        log_raw(indent_string() + "[Picard Iteration Elapsed: " + format_duration(elapsed) + "]");
        decrease_indent();
    }

    void log_discretization_start() {
        start_sub_stage();
        log_raw(indent_string() + "[Discretization] Start assembling system");
    }

    void log_discretization_end() {
        end_sub_stage("Assembling system");
    }

    void log_krylov_start() {
        start_sub_stage();
        log_raw(indent_string() + "[Linear Solver] Start Krylov solver");
    }

    void log_krylov_end(int iterations, double residual) {
        log_raw(indent_string() + "    Krylov Iterations :  " + std::to_string(iterations));
        log_raw(indent_string() + "    Krylov Residual   :  " + format_scientific(residual));
        end_sub_stage("Krylov solver");
        log_raw(indent_string() + "[Linear Solver] Finished");
    }

    void log_krylov_fallback_start() {
        start_sub_stage();
        log_raw(indent_string() + "[Fallback] Start SparseLU solver");
    }

    void log_krylov_fallback_end() {
        end_sub_stage("SparseLU solver");
        log_raw(indent_string() + "[Fallback] Finished SparseLU");
    }



    void log_nonlinear_residual(double norm_NUk, double deltaU) {
        log_raw(indent_string() + "[Nonlinear Residual]");
        log_raw(indent_string() + "    ||N(U_k)||         :  " + format_scientific(norm_NUk));
        log_raw(indent_string() + "    ||ΔU||             :  " + format_scientific(deltaU));
    }

    void log_convergence(bool deltaU_tol, bool rel_deltaU_tol, bool NUk_tol, bool rel_NUk_tol) {
        log_raw(indent_string() + "[Convergence Criteria]:");
        log_raw(indent_string() + "    ||ΔU|| < 1e-8            : " + (deltaU_tol ? "true" : "false"));
        log_raw(indent_string() + "    Rel(ΔU) < 1e-8           : " + (rel_deltaU_tol ? "true" : "false"));
        log_raw(indent_string() + "    ||N(U_k)|| < 1e-8        : " + (NUk_tol ? "true" : "false"));
        log_raw(indent_string() + "    Rel(||N(U_k)||) < 1e-8   : " + (rel_NUk_tol ? "true" : "false"));
    }

    void log_convergence_check(double delta, double rel_delta,
                            double res_norm, double rel_res_norm,
                            double rate_res_norm) 
    {
        log_raw(indent_string() + "[Convergence Check]");
        log_raw(indent_string() + "    delta            :  " + format_scientific(delta));
        log_raw(indent_string() + "    rel_delta        :  " + format_scientific(rel_delta));
        log_raw(indent_string() + "    res_norm         :  " + format_scientific(res_norm));
        log_raw(indent_string() + "    rel_res_norm     :  " + format_scientific(rel_res_norm));
        log_raw(indent_string() + "    rate_res_norm    :  " + format_scientific(rate_res_norm-1) + " + 1.0");
    }


    void log_summary(int steps, int nonlinear, int linear, const std::string& total_wall_time = "") {
        log_boxed_title("Simulation Finished");
        log_kv("Total Time Steps", steps);
        log_kv("Total Nonlinear Iters", nonlinear);
        log_kv("Total Linear Iters", linear);
        if (total_wall_time.empty()) {
            auto elapsed = elapsed_ms(start_time_);
            log_kv("Total Wall Time", format_duration(elapsed));
        } else {
            log_kv("Total Wall Time", total_wall_time);
        }
    }



    void set_indent(int n) { indent_ = n; }

private:
    const FilesystemManager& fsm_;
    std::string log_file_;
    std::ofstream log_stream_;


    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point stage_start_;
    std::chrono::steady_clock::time_point step_start_;
    std::chrono::steady_clock::time_point sub_stage_start_;
    std::chrono::steady_clock::time_point picard_start_;
    std::string stage_name_;
    int indent_;

    // 工具函数

    void log_raw(const std::string& line) {
        std::cout << line << std::endl;
        if (log_stream_.is_open())
            log_stream_ << line << std::endl;
    }

    void log_timed(const std::string& message) {
        auto elapsed = elapsed_ms(start_time_);
        std::stringstream ss;
        ss << "[" << format_duration(elapsed) << "] " << message;
        log_raw(ss.str());
    }
    template<typename T>
    void log_kv(const std::string& key, const T& value) {
        std::stringstream ss;
        ss << "  " << std::setw(15) << std::left << key << ":  " << value;
        log_raw(ss.str());
    }

    void log_boxed_line(const std::string& content) {
        std::string line = "════════════════════════════════════════════════════════════════════════";
        log_raw("╔" + line + "╗");
        log_raw("║ " + pad_text(content, 74) + " ║");
        log_raw("╚" + line + "╝");
    }

    void log_append_to_last_box(const std::string& extra) {
        // 这里可以根据实际情况做文件内容修改或下一行补充显示
        log_raw("  │" + extra);
    }

    void start_sub_stage() { sub_stage_start_ = std::chrono::steady_clock::now(); }

    void end_sub_stage(const std::string& task_name) {
        auto elapsed = elapsed_ms(sub_stage_start_);
        log_raw(indent_string() + "    [" + task_name + " elapsed: " + std::to_string(elapsed) + " ms]");
    }

    std::string indent_string() const {
        return std::string(indent_ * 4, ' ');
    }

    void increase_indent() { indent_++; }
    void decrease_indent() { indent_--; }

    long long elapsed_ms(const std::chrono::steady_clock::time_point& t0) const {
        using namespace std::chrono;
        auto now = steady_clock::now();
        return duration_cast<milliseconds>(now - t0).count();
    }

    std::string format_duration(long long ms) const {
        long long sec = ms / 1000;
        long long min = sec / 60;
        sec = sec % 60;
        ms = ms % 1000;
        std::stringstream ss;
        if (min > 0) ss << min << " min ";
        ss << sec << "." << std::setfill('0') << std::setw(3) << ms << " sec";
        return ss.str();
    }

    std::string format_scientific(double value) const {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3) << value;
        return oss.str();
    }

    std::string center_text(const std::string& text, size_t width) const {
        size_t left = (width - text.length()) / 2;
        size_t right = width - text.length() - left;
        return std::string(left, ' ') + text + std::string(right, ' ');
    }

    std::string pad_text(const std::string& text, size_t width) const {
        if (text.length() >= width) return text.substr(0, width);
        return text + std::string(width - text.length(), ' ');
    }
};
