#include "dg/dg_limiters/weno_limiters/pweight_weno_limiter_gpu.cuh"

// using namespace std;

// CUDA Kernel 实现

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void p_weight_weno_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const vector3f* d_points,
    DenseMatrix<5 * NumBasis, 1>* U_current)
{
    uInt cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id >= num_cells) return;

    const GPUTetrahedron& cell = d_cells[cell_id];
    const DenseMatrix<3,3>& J_self = cell.JacMat;
    const DenseMatrix<3,3>& JinvT_self = cell.invJac.transpose();

    const DenseMatrix<5 * NumBasis, 1>& coef_self = U_current[cell_id];
    DenseMatrix<5 * NumBasis, 1> coef_limited = coef_self;

    constexpr auto quad_pts = QuadC::get_points();
    constexpr auto quad_weights = QuadC::get_weights();

    std::array<std::array<Scalar, NumBasis>, QuadC::num_points> phi_table;
    for (uInt q = 0; q < QuadC::num_points; ++q)
        phi_table[q] = DGBasisEvaluator<Order>::eval_all(
            quad_pts[q][0], quad_pts[q][1], quad_pts[q][2]);
    std::array<std::array<vector3f,NumBasis>,QuadC::num_points> grads_table;
    for (uInt q = 0; q < QuadC::num_points; ++q) 
        grads_table[q] = DGBasisEvaluator<Order>::grad_all(
            quad_pts[q][0], quad_pts[q][1], quad_pts[q][2]);
    // if (cell_id == 42) {
    //     for(uInt idx = 0; idx < NumBasis; idx++){
    //         printf("coef_self: %.2e %.2e %.2e %.2e %.2e\n",
    //             coef_self[idx+0], coef_self[idx+1], coef_self[idx+2], coef_self[idx+3], coef_self[idx+4]);
    //     }
    // }
    
    for (int var = 0; var < 5; ++var)
    {
        Scalar avg_self = coef_self(5 * 0 + var, 0);

        DenseMatrix<NumBasis, 1> poly_self;
        for (uInt l = 0; l < NumBasis; ++l)
            poly_self(l, 0) = coef_self(5 * l + var, 0);
        // if (cell_id == 42) {
        //     printf("poly_self: ");
        //     for(uInt idx = 0; idx < NumBasis; idx++){
        //         printf("%.2e  ",
        //             poly_self[idx]);
        //     }
        //     printf("\n");
        // }
        DenseMatrix<NumBasis, 1> candidates[5];
        candidates[0] = poly_self;

        // if (cell_id == 42) {
        //     printf("candidates[0]: ");
        //     for(uInt idx = 0; idx < NumBasis; idx++){
        //         printf("%.2e  ",
        //             candidates[0][idx]);
        //     }
        //     printf("\n");
        // }
        Scalar weights[5] = {1e-16, 1e-16, 1e-16, 1e-16, 1e-16};

        // === 自身震荡指标 ===
        Scalar osc_self = 0;
        DenseMatrix<3,1> g_self = DenseMatrix<3,1>::Zeros();
        for (uInt l = 1; l < 4; ++l)
            g_self += poly_self(l, 0) * JinvT_self.multiply(grads_table[0][l]);
        osc_self = g_self.norm2();
        weights[0] = (200.0) / (1e-6 + osc_self);

        // === 邻居线性重构候选 ===
        for (int nf = 0; nf < 4; ++nf)
        {
            uInt nid = cell.neighbor_cells[nf];
            if (nid == uInt(-1)) continue;

            const GPUTetrahedron& ncell = d_cells[nid];
            const DenseMatrix<3,3>& J_n = ncell.JacMat;
            const DenseMatrix<3,3>& JinvT_n = ncell.invJac.transpose();
            const DenseMatrix<5 * NumBasis, 1>& coef_n = U_current[nid];

            DenseMatrix<3, 3> M = DenseMatrix<3, 3>::Zeros();
            DenseMatrix<3, 1> b = DenseMatrix<3, 1>::Zeros();

            for (uInt q = 0; q < QuadC::num_points; ++q)
            {
                DenseMatrix<3, 1> xg_ref = {quad_pts[q][0], quad_pts[q][1], quad_pts[q][2]};
                DenseMatrix<3, 1> phys_x = J_n.multiply(xg_ref) + d_points[ncell.nodes[0]];
                DenseMatrix<3, 1> ref_self = JinvT_self.multiply(phys_x - d_points[cell.nodes[0]]);

                const auto& phi_self = DGBasisEvaluator<Order>::eval_all(
                    ref_self[0], ref_self[1], ref_self[2]);
                


                // neighbor 解值
                Scalar uval = 0;
                for (uInt l = 0; l < NumBasis; ++l)
                    uval += phi_table[q][l] * coef_n(5 * l + var, 0);

                Scalar delta_u = uval - avg_self;
                Scalar w = quad_weights[q];

                for (int i = 0; i < 3; ++i)
                {
                    b(i, 0) += w * phi_self[i + 1] * delta_u;
                    for (int j = 0; j < 3; ++j)
                        M(i, j) += w * phi_self[i + 1] * phi_self[j + 1];
                }
            }

            DenseMatrix<3, 1> grad = M.solve(b);

            DenseMatrix<NumBasis, 1> poly_linear = DenseMatrix<NumBasis, 1>::Zeros();
            poly_linear(0, 0) = avg_self;
            for (int i = 0; i < 3; ++i)
                poly_linear(i + 1, 0) = grad(i, 0);
            candidates[nf + 1] = poly_linear;
            
            // if (cell_id == 42) {
            //     printf("candidates[%d + 1]: ",nf);
            //     for(uInt idx = 0; idx < NumBasis; idx++){
            //         printf("%.2e  ",
            //             candidates[nf + 1][idx]);
            //     }
            //     printf("\n");
            //     for(uInt row = 0; row<3; row++){
            //         for(uInt col = 0; col<3;col++){
            //             printf("M[%ld][%ld] = %lf \t \t ",row,col,M(row,col));
            //         }
            //         printf("\n");
            //     }
            // }

            Scalar osc = 0;
            // for (int i = 1; i < 4; ++i)
            //     osc += (poly_linear(i, 0) * JinvT_self.multiply(grads_table[0][i])).norm(); //poly_linear(i, 0) * poly_linear(i, 0);
            DenseMatrix<3,1> g_linear = DenseMatrix<3,1>::Zeros();
            for (uInt l = 1; l < 4; ++l)
                g_linear += poly_linear(l, 0) * JinvT_self.multiply(grads_table[0][l]);
            osc = g_linear.norm2();
            weights[nf + 1] = 1.0 / (1e-6 + osc);
        }

        // === 非线性加权平均 ===
        Scalar sum_w = 0;
        for (int k = 0; k < 5; ++k)
            sum_w += weights[k];

        // if (cell_id == 42) {
        //     printf("weights: %.2e %.2e %.2e %.2e %.2e\n",
        //         weights[0], weights[1], weights[2], weights[3], weights[4]);
        // }

        for (uInt l = 0; l < NumBasis; ++l)
        {
            Scalar uval = 0;
            for (int k = 0; k < 5; ++k)
                uval += weights[k] * candidates[k](l, 0);
            coef_limited(5 * l + var, 0) = uval / sum_w;
        }

        coef_limited(5 * 0 + var, 0) = avg_self; // 保平均值
    }

    U_current[cell_id] = coef_limited;
}


template<uInt Order, typename QuadC, typename QuadF>
void PWeightWENOLimiterGPU<Order, QuadC, QuadF>::apply(LongVectorDevice<5 * NumBasis>& current_coeffs)
{
    dim3 block(256);
    dim3 grid((mesh_.num_cells() + block.x - 1) / block.x);
    p_weight_weno_kernel<Order, NumBasis, QuadC, QuadF><<<grid, block>>>(
        mesh_.device_cells(), mesh_.num_cells(),
        mesh_.device_points(),
        current_coeffs.d_blocks);
}
