#pragma once
#include "base/type.h"
#include "base/exact.h"
#include "mesh/mesh.h"

// =====================
// 四象限 Riemann 问题：初值 + 边界条件统一函数
// =====================


constexpr Scalar param_gamma = 1.4;

// 给定四个区块的状态
struct State {
    Scalar rho, u, v, p;
};

struct WaveSpeeds {
    Scalar mass, momentum, energy;
};

// Compute 1D RH wave speed in normal direction between two states
__device__  constexpr  inline WaveSpeeds compute_wave_speeds_x(const State& L, const State& R) {
    WaveSpeeds s{0.0,0.0,0.0};
    s.mass = (R.rho * R.u - L.rho * L.u) / (R.rho - L.rho);
    s.momentum = ((R.rho * R.u * R.u + R.p) - (L.rho * L.u * L.u + L.p)) / (R.rho * R.u - L.rho * L.u);
    Scalar EL = (L.p / (param_gamma - 1.0) + 0.5 * L.rho * (L.u * L.u + L.v * L.v));// / L.rho;
    Scalar ER = (R.p / (param_gamma - 1.0) + 0.5 * R.rho * (R.u * R.u + R.v * R.v));// / R.rho;
    s.energy = (R.u * (ER + R.p) - L.u * (EL + L.p)) / (ER - EL);
    return s;
}

__device__  constexpr  inline WaveSpeeds compute_wave_speeds_y(const State& L, const State& R) {
    WaveSpeeds s{0.0,0.0,0.0};
    s.mass = (R.rho * R.v - L.rho * L.v) / (R.rho - L.rho);
    s.momentum = ((R.rho * R.v * R.v + R.p) - (L.rho * L.v * L.v + L.p)) / (R.rho * R.v - L.rho * L.v);
    Scalar EL = (L.p / (param_gamma - 1.0) + 0.5 * L.rho * (L.u * L.u + L.v * L.v));// / L.rho;
    Scalar ER = (R.p / (param_gamma - 1.0) + 0.5 * R.rho * (R.u * R.u + R.v * R.v));// / R.rho;
    s.energy = (R.v * (ER + R.p) - L.v * (EL + L.p)) / (ER - EL);
    return s;
}

// 四个象限初始状态
__device__ __constant__  constexpr State s1 = {1.5, 0.0,   0.0,   1.5};
__device__ __constant__  constexpr State s2 = {0.5323, 1.206, 0.0,   0.3};
__device__ __constant__  constexpr State s3 = {0.138,  1.206, 1.206, 0.029};
__device__ __constant__  constexpr State s4 = {0.5323, 0.0,   1.206, 0.3};
constexpr vector2f init_x0y0 = {0.80, 0.80};

// __device__ __constant__  constexpr State s1 = {  1.00,  0.75, -0.50,  1.00};
// __device__ __constant__  constexpr State s2 = {  2.00,  0.75,  0.50,  1.00};
// __device__ __constant__  constexpr State s3 = {  1.00, -0.75,  0.50,  1.00};
// __device__ __constant__  constexpr State s4 = {  3.00, -0.75, -0.50,  1.00};
// constexpr vector2f init_x0y0 = {0.50, 0.50};


// constexpr vector2f init_x0y0 = {0.80, 0.80};
// constexpr vector4f init_rho  = {1.5, 0.5323, 0.138, 0.5323};
// constexpr vector4f init_u    = {0.0, 1.206, 1.206, 0.0};
// constexpr vector4f init_v    = {0.0, 0.0, 1.206, 1.206};
// constexpr vector4f init_p    = {1.5, 0.3, 0.029, 0.3};


// constexpr vector2f init_x0y0 = {0.50, 0.50};
// constexpr vector4f init_rho  = {  1.00,  2.00,  1.00,  3.00};
// constexpr vector4f init_u    = {  0.75,  0.75, -0.75, -0.75};
// constexpr vector4f init_v    = { -0.50,  0.50,  0.50, -0.50};
// constexpr vector4f init_p    = {  1.00,  1.00,  1.00,  1.00};
