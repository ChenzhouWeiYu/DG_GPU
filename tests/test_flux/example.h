// 配置物理参数
ConvectiveRuntimeFlux::configure(1.4); // γ=1.4
RiemannSolvers::setType(RiemannSolvers::Roe);

// 运行计算
ImplicitConvectionRuntime<3>::assemble(mesh, U, mat, rhs);

// 动态修改参数（如模拟不同气体）
if (switch_to_argon) {
    ConvectiveRuntimeFlux::configure(5.0/3.0); // 单原子气体γ=5/3
    RiemannSolvers::setType(RiemannSolvers::HLLC);
}