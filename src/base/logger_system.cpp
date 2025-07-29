#include "base/logger_system.h"

std::string LoggerSystem::current_time_string() const {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%F %T", &tm_now);  // "YYYY-MM-DD HH:MM:SS"
    return std::string(buffer);
}


void LoggerSystem::start_step_timer() {
    step_start_ = std::chrono::steady_clock::now();
}

void LoggerSystem::end_step_timer_and_update() {
    auto now = std::chrono::steady_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(now - step_start_).count();

    if (avg_step_time_ms_ == 0.0) {
        avg_step_time_ms_ = duration_ms;
    } else {
        avg_step_time_ms_ = moving_avg_gamma_ * avg_step_time_ms_ + (1 - moving_avg_gamma_) * duration_ms;
    }
}

// ──────────────── 按步数保存 ────────────────
bool LoggerSystem::log_explicit_step(uInt step, Scalar t_current, Scalar dt, uInt next_save_step) {
    if(step == uInt(-1)) {
        start_step_timer();  // 第一次调用前的初始化，启动计时器
        avg_step_time_ms_ = 0.0;  // 重置平均时间
        moving_avg_gamma_ = 0.99;  // 重置移动平均系数
        return false;  // 第一次调用不需要保存
    }
    end_step_timer_and_update();  // 更新耗时统计

    // 节流打印（如每 100 步）
    // if (step % 100 == 0 || step == 0) 
    {
        std::stringstream ss;
        ss <<  "[" + current_time_string() + "] " ;
        ss << "[Time Step] Step: " << std::setw(7) << std::right << step
           << " │ Simulated Time: " << std::fixed << std::setprecision(6) << t_current << " s"
           << " │ Δt = " << dt << " s"
           << " │ Avg. Step Time: " << format_duration(avg_step_time_ms_);

        // 如果设置了 next_save_step，显示预估 Wall 时间
        if (next_save_step > step) {
            double steps_left = next_save_step - step;
            double est_wall_ms = steps_left * avg_step_time_ms_;

            ss << " │ Est. Next Save Step: " << std::setw(7) << std::right << next_save_step
               << " │ Est. Wall Time Left: " << format_duration(est_wall_ms);
        }

        log_raw(ss.str());
    }

    start_step_timer();  // 启动下一个 step 的计时器

    return step >= next_save_step;
}

// ──────────────── 按时间点保存 ────────────────
bool LoggerSystem::log_explicit_step(uInt step, Scalar t_current, Scalar dt, Scalar next_save_time) {
    if(step == uInt(-1)) {
        start_step_timer();  // 第一次调用前的初始化，启动计时器
        avg_step_time_ms_ = 0.0;  // 重置平均时间
        moving_avg_gamma_ = 0.95;  // 重置移动平均系数
        return false;  // 第一次调用不需要保存
    }
    end_step_timer_and_update();  // 更新耗时统计

    // 节流打印
    // if (step % 100 == 0 || step == 0) 
    {
        std::stringstream ss;
        ss <<  "[" + current_time_string() + "] " ;
        ss << "[Time Step] Step: " << std::setw(7) << std::right << step
           << " │ Simulated Time: " << std::fixed << std::setprecision(6) << t_current << " s"
           << " │ Δt = " << dt << " s"
           << " │ Avg. Step Time: " << format_duration(avg_step_time_ms_);

        if (t_current < next_save_time) {
            double time_left = next_save_time - t_current;
            double steps_left = time_left / dt;
            double est_wall_ms = steps_left * avg_step_time_ms_;

            ss << " │ Est. Next Save Time: " << std::setprecision(6) << next_save_time << " s"
               << " │ Est. Wall Time Left: " << format_duration(est_wall_ms);
        }

        log_raw(ss.str());
    }

    start_step_timer();  // 启动下一个 step 的计时器

    return t_current >= next_save_time-1e-12;
}

void LoggerSystem::log_save_solution(uInt step, Scalar t_current, const std::string& filename) {
    std::stringstream ss;
    ss <<  "[" + current_time_string() + "] " ;
    ss << "[Save] Saving solution at Step " << std::setw(7) << std::right << step
       << " │ Simulated Time: " << std::fixed << std::setprecision(6) << t_current << " s"
       << " │ File: " << filename;

    log_raw(ss.str());
}



// ──────────────── 样式功能 ─────────────────
void LoggerSystem::log_boxed_title(const std::string& title) {
    std::string line = "══════════════════════════════════════════════════════════════════════════";
    log_raw("╔" + line + "╗");
    log_raw("║" + center_text( "[" + current_time_string() + "] " + title, 74) + "║");
    log_raw("╚" + line + "╝");
}

void LoggerSystem::log_section_title(const std::string& section) {
    log_raw("");
    log_boxed_title(section);
}

void LoggerSystem::print_header(const std::string& title) {
    log_raw("");
    log_boxed_title(title);
}

void LoggerSystem::print_config(int order, int meshN, int cpus) {
    log_kv("Order", order);
    log_kv("Mesh N", meshN);
    log_kv("CPU Cores", cpus);
    log_kv("Solution File", fsm_.get_solution_file(0, meshN));
    log_kv("Error Log", fsm_.get_error_log_file());
    log_kv("Config File", fsm_.get_config_file());
    log_kv("Run Info", fsm_.get_run_info_file());
}

void LoggerSystem::print_mesh_info(const ComputingMesh& cpu_mesh) {
    log_raw("Mesh Information:");
    log_kv("Number of Points", cpu_mesh.m_points.size());
    log_kv("Number of Faces", cpu_mesh.m_faces.size());
    log_kv("Number of Cells", cpu_mesh.m_cells.size());
}

void LoggerSystem::print_mesh_info(const DeviceMesh& gpu_mesh) {
    log_raw("Mesh Information:");
    log_kv("Number of Points", gpu_mesh.num_points());
    log_kv("Number of Faces", gpu_mesh.num_faces());
    log_kv("Number of Cells", gpu_mesh.num_cells());
    log_kv("Device Memory Usage (MB)", gpu_mesh.get_memory_usage());
}
// ──────────────── 计时功能 ─────────────────

void LoggerSystem::start_stage(const std::string& stage_name) {
    stage_name_ = stage_name;
    stage_start_ = std::chrono::steady_clock::now();
    log_timed( "[" + current_time_string() + "] " + "Start " + stage_name);
}

void LoggerSystem::end_stage() {
    auto elapsed = elapsed_ms(stage_start_);
    log_timed( "[" + current_time_string() + "] " + "Finished " + stage_name_ + "  (elapsed: " + std::to_string(elapsed) + " ms)");
}

void LoggerSystem::log_time_step(int step, double t, double dt) {
    step_start_ = std::chrono::steady_clock::now();
    std::stringstream ss;
    ss << "Time Step " << step
    << " │ t = " << std::fixed << std::setprecision(6) << t
    << " sec │ Δt = " << std::setprecision(6) << dt
    << " sec";
    log_boxed_line(ss.str());
}

void LoggerSystem::end_time_step() {
    auto elapsed = elapsed_ms(step_start_);
    log_append_to_last_box(" Elapsed: " + format_duration(elapsed));
}

void LoggerSystem::log_picard_iteration(int iter) {
    picard_start_ = std::chrono::steady_clock::now();
    log_raw(indent_string() + "── Picard Iteration " + std::to_string(iter) + " ────────────────────────────────────────");
    increase_indent();
}

void LoggerSystem::end_picard_iteration() {
    auto elapsed = elapsed_ms(picard_start_);
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "[Picard Iteration Elapsed: " + format_duration(elapsed) + "]");
    decrease_indent();
}

void LoggerSystem::log_discretization_start() {
    start_sub_stage();
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "[Discretization] Start assembling system");
}

void LoggerSystem::log_discretization_end() {
    end_sub_stage("Assembling system");
}

void LoggerSystem::log_krylov_start() {
    start_sub_stage();
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "[Linear Solver] Start Krylov solver");
}

void LoggerSystem::log_krylov_end(int iterations, double residual) {
    log_raw(indent_string() + "    Krylov Iterations :  " + std::to_string(iterations));
    log_raw(indent_string() + "    Krylov Residual   :  " + format_scientific(residual));
    end_sub_stage("Krylov solver");
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "[Linear Solver] Finished");
}

void LoggerSystem::log_krylov_fallback_start() {
    start_sub_stage();
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "[Fallback] Start SparseLU solver");
}

void LoggerSystem::log_krylov_fallback_end() {
    end_sub_stage("SparseLU solver");
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "[Fallback] Finished SparseLU");
}



void LoggerSystem::log_nonlinear_residual(double norm_NUk, double deltaU) {
    log_raw(indent_string() + "[Nonlinear Residual]");
    log_raw(indent_string() + "    ||N(U_k)||         :  " + format_scientific(norm_NUk));
    log_raw(indent_string() + "    ||ΔU||             :  " + format_scientific(deltaU));
}

void LoggerSystem::log_convergence(bool deltaU_tol, bool rel_deltaU_tol, bool NUk_tol, bool rel_NUk_tol) {
    log_raw(indent_string() + "[Convergence Criteria]:");
    log_raw(indent_string() + "    ||ΔU|| < 1e-8            : " + (deltaU_tol ? "true" : "false"));
    log_raw(indent_string() + "    Rel(ΔU) < 1e-8           : " + (rel_deltaU_tol ? "true" : "false"));
    log_raw(indent_string() + "    ||N(U_k)|| < 1e-8        : " + (NUk_tol ? "true" : "false"));
    log_raw(indent_string() + "    Rel(||N(U_k)||) < 1e-8   : " + (rel_NUk_tol ? "true" : "false"));
}

void LoggerSystem::log_convergence_check(double delta, double rel_delta,
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


void LoggerSystem::log_summary(int steps, int nonlinear, int linear, const std::string& total_wall_time ) {
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















// 工具函数


void LoggerSystem::log_raw(const std::string& line) {
    std::cout << line << std::endl;
    if (log_stream_.is_open())
        log_stream_ << line << std::endl;
}

void LoggerSystem::log_timed(const std::string& message) {
    auto elapsed = elapsed_ms(start_time_);
    std::stringstream ss;
    ss << "[" << format_duration(elapsed) << "] " << message;
    log_raw(ss.str());
}
template<typename T>
void LoggerSystem::log_kv(const std::string& key, const T& value) {
    std::stringstream ss;
    ss << "  " << std::setw(15) << std::left << key << ":  " << value;
    log_raw(ss.str());
}

void LoggerSystem::log_boxed_line(const std::string& content) {
    std::string line = "════════════════════════════════════════════════════════════════════════";
    log_raw("╔" + line + "╗");
    log_raw("║ " + pad_text( "[" + current_time_string() + "] " + content, 74) + " ║");
    log_raw("╚" + line + "╝");
}

void LoggerSystem::log_append_to_last_box(const std::string& extra) {
    // 这里可以根据实际情况做文件内容修改或下一行补充显示
    log_raw("  │" + extra);
}

void LoggerSystem::start_sub_stage() { sub_stage_start_ = std::chrono::steady_clock::now(); }

void LoggerSystem::end_sub_stage(const std::string& task_name) {
    auto elapsed = elapsed_ms(sub_stage_start_);
    log_raw(indent_string() +  "[" + current_time_string() + "] " + "    [" + task_name + " elapsed: " + std::to_string(elapsed) + " ms]");
}

std::string LoggerSystem::indent_string() const {
    return std::string(indent_ * 4, ' ');
}

void LoggerSystem::increase_indent() { indent_++; }
void LoggerSystem::decrease_indent() { indent_--; }

long long LoggerSystem::elapsed_ms(const std::chrono::steady_clock::time_point& t0) const {
    using namespace std::chrono;
    auto now = steady_clock::now();
    return duration_cast<milliseconds>(now - t0).count();
}

std::string LoggerSystem::format_duration(long long ms) const {
    long long sec = ms / 1000;
    long long min = sec / 60;
    sec = sec % 60;
    ms = ms % 1000;
    std::stringstream ss;
    if (min > 0) ss << min << " min " << sec << "." << std::setfill('0') << std::setw(3) << ms << " sec";
    else if (sec > 0) ss << sec << " s " << std::setw(3) << ms << " ms";
    else ss << ms << " ms";
    return ss.str();
}

std::string LoggerSystem::format_scientific(double value) const {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3) << value;
    return oss.str();
}

std::string LoggerSystem::center_text(const std::string& text, size_t width) const {
    size_t left = (width - text.length()) / 2;
    size_t right = width - text.length() - left;
    return std::string(left, ' ') + text + std::string(right, ' ');
}

std::string LoggerSystem::pad_text(const std::string& text, size_t width) const {
    if (text.length() >= width) return text.substr(0, width);
    return text + std::string(width - text.length(), ' ');
}