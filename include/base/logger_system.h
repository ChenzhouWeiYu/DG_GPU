#pragma once

#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <filesystem>
#include "base/filesystem_manager.h"
#include "mesh/computing_mesh.h"
#include "mesh/device_mesh.h"

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
    void log_boxed_title(const std::string& title) ;

    void log_section_title(const std::string& section) ;

    void print_header(const std::string& title) ;

    void print_config(int order, int meshN, int cpus) ;

    // ──────────────── 计时功能 ─────────────────

    void start_stage(const std::string& stage_name) ;

    void end_stage() ;

    void log_time_step(int step, double t, double dt) ;

    void end_time_step() ;

    void log_picard_iteration(int iter) ;

    void end_picard_iteration() ;

    void log_discretization_start() ;

    void log_discretization_end() ;

    void log_krylov_start() ;

    void log_krylov_end(int iterations, double residual) ;

    void log_krylov_fallback_start() ;

    void log_krylov_fallback_end() ;

    std::string current_time_string() const ;

    void log_nonlinear_residual(double norm_NUk, double deltaU) ;

    void log_convergence(bool deltaU_tol, bool rel_deltaU_tol, bool NUk_tol, bool rel_NUk_tol) ;

    void log_convergence_check(double delta, double rel_delta,
                            double res_norm, double rel_res_norm,
                            double rate_res_norm) ;


    void log_summary(int steps, int nonlinear, int linear, const std::string& total_wall_time = "") ;

    void print_mesh_info(const ComputingMesh& cpu_mesh) ;
    void print_mesh_info(const DeviceMesh& gpu_mesh) ;

    void set_indent(int n) { indent_ = n; }

    // 按固定步数保存
    bool log_explicit_step(uInt step, Scalar t_current, Scalar dt, uInt next_save_step);

    // 按固定时间点保存
    bool log_explicit_step(uInt step, Scalar t_current, Scalar dt, Scalar next_save_time);

    // 保存时输出日志
    void log_save_solution(uInt step, Scalar t_current, const std::string& filename);
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

    // 用于显格式的输出和保存
    void start_step_timer();
    void end_step_timer_and_update();

    // std::chrono::steady_clock::time_point step_start_;
    double avg_step_time_ms_ = 0.0;
    double moving_avg_gamma_ = 0.95;  // 移动平均系数
    // 结束

    // 工具函数

    void log_raw(const std::string& line) ;

    void log_timed(const std::string& message) ;
    template<typename T>
    void log_kv(const std::string& key, const T& value) ;

    void log_boxed_line(const std::string& content) ;

    void log_append_to_last_box(const std::string& extra) ;

    void start_sub_stage() ;

    void end_sub_stage(const std::string& task_name) ;

    std::string indent_string() const ;

    void increase_indent() ;
    void decrease_indent() ;

    long long elapsed_ms(const std::chrono::steady_clock::time_point& t0) const ;

    std::string format_duration(long long ms) const ;

    std::string format_scientific(double value) const ;

    std::string center_text(const std::string& text, size_t width) const ;

    std::string pad_text(const std::string& text, size_t width) const ;
};
