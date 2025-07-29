// FilesystemManager.cpp
#include "base/filesystem_manager.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

FilesystemManager::FilesystemManager(const std::string& base_dir)
    : base_dir_(base_dir) {}

std::string FilesystemManager::get_solution_file(int time_step, int mesh_size) const {
    return base_dir_ + "/solution/T_" + std::to_string(time_step)
           + "_N_" + std::to_string(mesh_size) + ".txt";
}
std::string FilesystemManager::get_solution_file_vtk(int time_step, int mesh_size) const {
    return base_dir_ + "/solution/T_" + std::to_string(time_step)
           + "_N_" + std::to_string(mesh_size) + ".vtk";
}
std::string FilesystemManager::get_solution_file_vtu(int time_step, int mesh_size) const {
    return base_dir_ + "/solution/T_" + std::to_string(time_step)
           + "_N_" + std::to_string(mesh_size) + ".vtu";
}
std::string FilesystemManager::get_solution_file_txt(int time_step, int mesh_size) const {
    return base_dir_ + "/solution/T_" + std::to_string(time_step)
           + "_N_" + std::to_string(mesh_size) + ".txt";
}
std::string FilesystemManager::get_solution_file_h5(int time_step, int mesh_size) const {
    return base_dir_ + "/solution/T_" + std::to_string(time_step)
           + "_N_" + std::to_string(mesh_size) + ".h5";
}

std::string FilesystemManager::get_error_log_file() const {
    return base_dir_ + "/error_log.txt";
}

std::string FilesystemManager::get_config_file() const {
    return base_dir_ + "/config.json";
}

std::string FilesystemManager::get_run_info_file() const {
    return base_dir_ + "/run_info.txt";
}

void FilesystemManager::prepare_output_directory() {
    fs::path path(base_dir_);

    // 如果已存在，备份
    if (fs::exists(path) && fs::is_directory(path)) {
        fs::rename(path, generate_backup_name());
    }

    // 创建新的目录和子目录
    fs::create_directories(path);
    fs::create_directories(path / "solution");
}

std::string FilesystemManager::generate_backup_name() const {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&now_c));
    return base_dir_ + "_backup_" + std::string(buffer);
}

