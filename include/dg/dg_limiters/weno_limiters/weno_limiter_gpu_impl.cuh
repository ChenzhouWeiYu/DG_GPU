#include "dg/dg_limiters/weno_limiters/weno_limiter_gpu.cuh"

// using namespace std;

// CUDA Kernel 实现

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void apply_weno_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells, 
    const vector3f* d_points, 
    DenseMatrix<5 * NumBasis, 1>* U_current)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    const GPUTetrahedron& cell = d_cells[cellId];
    // const DenseMatrix<3,3>& JacT_inv_self = cell.invJac;
    const DenseMatrix<3,3>& Jac_self = cell.JacMat;

    const DenseMatrix<5 * NumBasis, 1>& coef_self = U_current[cellId];

    // 限制后的新模态系数
    DenseMatrix<5 * NumBasis, 1> coef_limited = coef_self;

    constexpr auto basis_table = QuadC::get_points();
    constexpr auto weight_table = QuadC::get_weights();

    std::array<std::array<vector3f,NumBasis>,QuadC::num_points> grads_table;
    for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
        const auto& xg = basis_table[xgi];
        const auto& grads = DGBasisEvaluator<Order>::grad_all(xg[0], xg[1], xg[2]);
        grads_table[xgi] = grads;
    }

    for (int var = 0; var < 5; ++var) {
        // ------------------- Step 1: 获取自身均值 -------------------
        Scalar avg_self = coef_self(5*0 + var, 0);

        // ------------------- Step 2: 构造候选多项式 -------------------
        // 候选 0: 自身
        DenseMatrix<NumBasis,1> poly_self;
        for (uInt l = 0; l < NumBasis; ++l)
            poly_self(l,0) = coef_self(5*l + var, 0);

        // 候选 1-4: 邻居线性外插
        DenseMatrix<NumBasis,1> candidates[5];
        candidates[0] = poly_self;

        // 充分小，保证即使是缺少邻居，也不会造成影响
        Scalar weights[5] = {1e-16,1e-16,1e-16,1e-16,1e-16}; 

        // 你之前遗漏了自身的权重计算，并且看别人文章，自身要给一个足够大的权重？
        {
            Scalar osc = 0;
            for (uInt l = 1; l < 4; ++l)
                osc += poly_self(l,0)*poly_self(l,0);
            weights[0] = (1-4e-3) / (1e-16 + osc);
        }
        // {
        //     Scalar osc = 0;
        //     for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi){
        //         const Scalar w = weight_table[xgi] * cell.volume * 6;
        //         const auto& grads = grads_table[xgi];
        //         DenseMatrix<3,1> gradU = DenseMatrix<3,1>::Zeros();
        //         for (uInt l = 0; l < NumBasis; ++l) {
        //             // grad φ_l = [∂φ/∂ξ, ∂φ/∂η, ∂φ/∂ζ]
        //             // DenseMatrix<3,1> grad_phi_l(grads[l]);
        //             // const auto& phys_grad = cell.invJac.multiply(grad_phi_l);
        //             const auto& phys_grad = cell.invJac.multiply(grads[l]);
        //             gradU += poly_self(l,0) * phys_grad;
        //         }
        //         osc += w * gradU.norm2();
        //     }
        //     weights[0] = (1 - 4e-3) / (1e-16 + osc);
        // }
        
        for (int nf = 0; nf < 4; ++nf) {
            uInt nid = cell.neighbor_cells[nf];
            if (nid == uInt(-1)) continue;
            const GPUTetrahedron& ncell = d_cells[nid];
            const DenseMatrix<3,3>& JacT_inv_n = ncell.invJac;
            // const DenseMatrix<3,3>& Jac_n = ncell.JacMat;
            const DenseMatrix<5 * NumBasis, 1>& coef_n = U_current[nid];

            // Step 2.1: 构造邻居上的一阶模态 u(x) = a + b*x + c*y + d*z
            DenseMatrix<4,1> neighbor_poly;
            for (uInt l = 0; l < 4; ++l)
                neighbor_poly(l,0) = coef_n(5*l + var, 0);

            // Step 2.2: 将该多项式变换到当前单元的参考坐标上
            // 仿射变换: x_self = J_self * xi_self + x0_self
            //         = J_n * xi_n + x0_n
            // => xi_n = J_n^{-1} * (J_self * xi_self + x0_self - x0_n)
            // 只需要 xi_self -> x -> xi_n -> eval -> reproject

            // 通过积分和正规正交系数来重构目标单元上的模态系数
            // DenseMatrix<4,4> M = DenseMatrix<4,4>::Zeros();
            // 由于正交性，只需要计算对角元素
            DenseMatrix<4,1> diagM = DenseMatrix<4,1>::Zeros(); 
            DenseMatrix<4,1> b = DenseMatrix<4,1>::Zeros();
            for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
                const auto& xg = basis_table[xgi];
                const auto& basis_self = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
                const auto& phys_x = Jac_self.multiply(vector3f{xg[0], xg[1], xg[2]}) + d_points[cell.nodes[0]];
                const auto& ref_n = JacT_inv_n.transpose().multiply(phys_x - d_points[ncell.nodes[0]]);
                const auto& basis_n = DGBasisEvaluator<Order>::eval_all(ref_n[0], ref_n[1], ref_n[2]);

                Scalar uval = 0;
                for (uInt l = 0; l < 4; ++l)
                    uval += basis_n[l] * neighbor_poly(l,0);

                for (uInt i = 0; i < 4; ++i) {
                    b(i,0) += basis_self[i] * uval * weight_table[xgi];
                    // 由于正交性，只需要计算对角元素
                    diagM(i,0) += basis_self[i] * basis_self[i] * weight_table[xgi];
                    // for (uInt j = 0; j < 4; ++j)
                    //     M(i,j) += basis_self[i] * basis_self[j] * weight_table[i];
                }
            }
            const auto& poly_proj = b/diagM; //M.solve(b);
            for(uInt kk = 0; kk<4; kk++){
                candidates[nf+1][kk] = poly_proj[kk];
            }
            

            Scalar osc = 0;
            for (uInt l = 1; l < 4; ++l)
                osc += neighbor_poly(l,0)*neighbor_poly(l,0);
            weights[nf+1] = 1.0e-3 / (1e-16 + osc);
            // Scalar osc = 0;
            // for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi){
            //     const Scalar w = weight_table[xgi] * cell.volume * 6;
            //     const auto& grads = grads_table[xgi];
            //     DenseMatrix<3,1> gradU = DenseMatrix<3,1>::Zeros();
            //     for (uInt l = 0; l < 4; ++l) {
            //         // grad φ_l = [∂φ/∂ξ, ∂φ/∂η, ∂φ/∂ζ]
            //         // DenseMatrix<3,1> grad_phi_l(grads[l]);
            //         // const auto& phys_grad = cell.invJac.multiply(grad_phi_l);
            //         const auto& phys_grad = cell.invJac.multiply(grads[l]);
            //         gradU += poly_proj(l,0) * phys_grad;
            //     }
            //     osc += w * gradU.norm2();
            // }
            // weights[nf+1] = 1.0e-3 / (1e-16 + osc);
        }

        // ------------------- Step 3: 非线性加权 -------------------
        Scalar sumW = weights[0] + weights[1] + weights[2] + weights[3] + weights[4];
        // printf("%.4f,%.4f,%.4f,%.4f,%.4f",weights[0]/sumW,weights[1]/sumW,weights[2]/sumW,weights[3]/sumW,weights[4]/sumW);
        for (uInt l = 0; l < NumBasis; ++l) {
            coef_limited(5*l + var, 0) = 0;
            for (int k = 0; k < 5; ++k)
                coef_limited(5*l + var, 0) += weights[k] * candidates[k](l,0) / sumW;
        }
        coef_limited(5*0 + var, 0) = avg_self;
    }

    U_current[cellId] = coef_limited;
}

template<uInt Order, typename QuadC, typename QuadF>
void WENOLimiterGPU<Order, QuadC, QuadF>::apply(LongVectorDevice<5 * NumBasis>& current_coeffs)
{
    dim3 block(256);
    dim3 grid((mesh_.num_cells() + block.x - 1) / block.x);
    apply_weno_kernel<Order, NumBasis, QuadC, QuadF><<<grid, block>>>(
        mesh_.device_cells(), mesh_.num_cells(),
        mesh_.device_points(), 
        current_coeffs.d_blocks);
}
