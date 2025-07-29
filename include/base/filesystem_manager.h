// FilesystemManager.h
#pragma once
#include "base/type.h"

class FilesystemManager {
public:
    explicit FilesystemManager(const std::string& base_dir);

    // 获取 solution 文件路径
    std::string get_solution_file(int time_step, int mesh_size) const;
    std::string get_solution_file_txt(int time_step, int mesh_size) const;
    std::string get_solution_file_vtk(int time_step, int mesh_size) const;
    std::string get_solution_file_vtu(int time_step, int mesh_size) const;
    std::string get_solution_file_h5(int time_step, int mesh_size) const;

    // 获取 error 日志文件路径
    std::string get_error_log_file() const;

    // 获取 config 文件路径
    std::string get_config_file() const;

    // 获取 run info 文件路径
    std::string get_run_info_file() const;

    // 初始化输出目录（备份旧目录 + 创建新目录）
    void prepare_output_directory();

private:
    std::string base_dir_;

    // 辅助函数：生成带时间戳的备份名
    std::string generate_backup_name() const;
};