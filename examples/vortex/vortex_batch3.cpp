#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
// #include "dg/dg_schemes/explicit_convection.h"
// #include "dg/dg_schemes/explicit_diffusion.h"
#include "dg/dg_schemes/implicit_convection.h"
// #include "dg/dg_schemes/implicit_diffusion.h"
#include "EigenSolver/EigenSparseSolver.h"
// #include "unsupported/Eigen/IterativeSolvers"

string sol_filename(uInt Order, uInt Time, uInt Nmesh){
    return "./Order_" + std::to_string(Order) 
            + "/solution/T_" + std::to_string(Time) 
            + "_N_" + std::to_string(Nmesh) + ".txt";
}

#include "problem.h"
ComputingMesh create_mesh(uInt N){
    GeneralMesh mesh = OrthHexMesh({0,0,0},{10,10,10.0/N},{N,N,1});
    mesh.split_hex5_scan();                                   
    mesh.rebuild_cell_topology();                             
    mesh.validate_mesh();                                     
    ComputingMesh cmesh(mesh);                                
    cmesh.m_boundaryTypes.resize(cmesh.m_faces.size());                   
    for(uInt faceId=0;faceId<cmesh.m_faces.size();faceId++){           
        if(cmesh.m_faces[faceId].m_neighbor_cells[1]==uInt(-1)){ 
            const auto& face = cmesh.m_faces[faceId];           
            if(std::abs(face.m_normal[2])>0.5 ) 
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            // else if(std::abs(face.m_normal[1])>0.5)          
            //     cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DY;
            else
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
        }
    }
    return cmesh;
}



template<uInt Order>
void Run(uInt N){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    auto logging = [&](std::string ss){debug("Time  " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    debug("Start   " + std::to_string(chrone_clock()));

    const auto& cmesh = create_mesh(N);
    logging("Split Hex Mesh to Tet");

    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 5*Basis::NumBasis;

    LongVector<DoFs> x(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        const auto& rhoU_coef = Basis::func2coef([&](vector3f Xi)->DenseMatrix<5,1>{
            return {rho_Xi(cell,Xi),rhou_Xi(cell,Xi),rhov_Xi(cell,Xi),rhow_Xi(cell,Xi),rhoe_Xi(cell,Xi)};
        });
        for(uInt k=0;k<Basis::NumBasis;k++){
            MatrixView<DoFs,1,5,1>(x[cellId],5*k,0) = rhoU_coef[k];
        }
    }
    
    logging("Set Init Value");

    ImplicitConvection<Basis::OrderBasis> convection;
    
    LongVector<DoFs> mass(x.size());
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        DenseMatrix<DoFs,DoFs> mass_matrix;
        for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
            const auto& p = QuadC::points[xgId];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    mass[cellId](5*i + k, 0) += phi[i] * phi[i] * 
                        QuadC::weights[xgId] * cmesh.m_cells[cellId].compute_jacobian_det();
                }
            }
        }
    }
    
    print(std::array<std::string,8>{"#       time", "rel.err  rho",
                    "rel.err  u", "rel.err  v", "rel.err  w", 
                    "rel.err  e", "rel.err coef", "cpu time"});
    for(uInt kkkk=0;kkkk<1;kkkk++){
        
        Scalar Dt = 0.5;
        Scalar max_dt = Dt * std::pow((5.0/N),(Order+1));
        Scalar dt = max_dt;
        Scalar sub_t = 0;
        do{ 
            dt = std::min(max_dt,Dt-sub_t);
            sub_t += dt;

            
            LongVector<DoFs> dx = x;

            for(uInt picard_iter = 0; picard_iter < 100; picard_iter++){

                LongVector<DoFs> rhs(x.size());
                BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                
                convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);
                
                for(uInt cellId = 0;cellId<cmesh.m_cells.size();cellId++){
                    const auto& diag = DenseMatrix<DoFs,DoFs>::Diag(mass[cellId]/dt);
                    sparse_mat.add_block(cellId, cellId, diag);
                    rhs[cellId] += diag.multiply(x[cellId]);
                }

                sparse_mat.finalize();

                EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
                LongVector<DoFs> ddx = solver.BiCGSTAB(dx);

                const auto& delta_x = ddx - dx;
                dx = ddx;
                Scalar delta = delta_x.dot(delta_x);
                delta = std::sqrt(delta);

                std::ostringstream oss;
                oss << delta;
                {
                    LongVector<DoFs> rhs(x.size());
                    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                    

                    // 用 保存的 dx 代入、离散
                    convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                        sparse_mat, rhs);
                    sparse_mat.finalize();
                    const auto& residual = (mass/dt*dx + sparse_mat.multiply(dx)) - (rhs + mass/dt * x);
                    std::ostringstream oss;
                    oss << std::sqrt(residual.dot(residual));
                    logging("Picard iter " + std::to_string(picard_iter) + "  " + oss.str());

                }
                logging("Picard iter " + std::to_string(picard_iter) + "  " + oss.str());
                if(delta < 1e-10) break;
            }
            // Picard 结束，赋值，x 为 u^{n+1}
            x = dx;

            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));

        }while(sub_t < Dt);


        Scalar curr_time = Dt * (kkkk+1);

        std::ofstream fp;
        fp.open(sol_filename(Basis::OrderBasis, kkkk+1, N));
        auto err_integral = [&](LongVector<5*QuadC::num_points> U_h,LongVector<5*QuadC::num_points> U_s){
            DenseMatrix<5,1> err_per_cells = DenseMatrix<5,1>::Zeros();
            DenseMatrix<5,1> sol_per_cells = 1e-47 * DenseMatrix<5,1>::Ones();
            for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
                const auto& cell = cmesh.m_cells[cellId];
                for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                    const DenseMatrix<5,1>& bUh = MatrixView<5*QuadC::num_points,1,5,1>(U_h[cellId],5*xgId,0);
                    const DenseMatrix<5,1>& bUs = MatrixView<5*QuadC::num_points,1,5,1>(U_s[cellId],5*xgId,0);
                    const DenseMatrix<5,1>& bUe = bUh - bUs;
                    const auto& weight = QuadC::weights[xgId] * cell.compute_jacobian_det();
                    // Scalar error_cell = uh[cellId][xgId] - us[cellId][xgId];

                    err_per_cells += pow(bUe,2) * weight;
                    sol_per_cells += pow(bUs,2) * weight;
                }
            }
            return pow(err_per_cells/sol_per_cells,0.5);
        };
        LongVector<5*QuadC::num_points> U_h(cmesh.m_cells.size());
        LongVector<5*QuadC::num_points> U_s(cmesh.m_cells.size());
        LongVector<DoFs> error(cmesh.m_cells.size());
        #pragma omp parallel for schedule(dynamic)
        for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
            const auto& cell = cmesh.m_cells[cellId];
            // rho[cellId] = 0.0;
            const auto& U_func = [&](vector3f Xi)->DenseMatrix<5,1>{
                return {
                    rho_Xi(cell,Xi,curr_time),
                    rhou_Xi(cell,Xi,curr_time),
                    rhov_Xi(cell,Xi,curr_time),
                    rhow_Xi(cell,Xi,curr_time),
                    rhoe_Xi(cell,Xi,curr_time)
                    };
            };
            const auto& U_coef = Basis::func2coef(U_func);
            for(uInt k=0;k<Basis::NumBasis;k++){
                const DenseMatrix<5,1>& block_coef = MatrixView<DoFs,1,5,1>(x[cellId],5*k,0) - U_coef[k];
                MatrixView<DoFs,1,5,1>(error[cellId],5*k,0) = block_coef;
            }
            for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                const auto& p = QuadC::points[xgId];
                const auto& pos = cell.transform_to_physical(p);
                const auto& value = Basis::eval_all(p[0],p[1],p[2]);
                const auto& U = Basis::template coef2filed<5,Scalar>(x[cellId],p);
                MatrixView<5*QuadC::num_points,1,5,1> block_U_h(U_h[cellId],5*xgId,0);
                MatrixView<5*QuadC::num_points,1,5,1> block_U_s(U_s[cellId],5*xgId,0);
                
                block_U_h = U;
                block_U_h[0] = U[0];
                block_U_s = U_func(p);
            }
        }

        for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
            const auto& cell = cmesh.m_cells[cellId];
            for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                const auto& p = QuadC::points[xgId];
                const auto& pos = cell.transform_to_physical(p);
                const auto& bUh = MatrixView<5*QuadC::num_points,1,5,1>(U_h[cellId],5*xgId,0);
                const auto& bUs = MatrixView<5*QuadC::num_points,1,5,1>(U_s[cellId],5*xgId,0);
                fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                 << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
                 << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
                 << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
                 << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] 
                 << "  " <<std::setprecision(16)<<  bUh[4] << "  " <<std::setprecision(16)<<  bUs[4] << std::endl;
            }
        }
        fp.close();

        const auto& U_err = err_integral(U_h,U_s);
        print(std::array<Scalar,8>{curr_time, U_err[0], U_err[1],U_err[2],U_err[3],U_err[4], 
                std::sqrt(error.dot(error)/x.dot(x)), chrone_clock()});
        // print(vector3f{curr_time, std::sqrt(err.dot(err)/rho.dot(rho)),chrone_clock()});
    }
}

// template void Run<0>(uInt);
// template void Run<1>(uInt);
// template void Run<2>(uInt);
// template void Run<3>(uInt);
// template void Run<4>(uInt);
// template void Run<5>(uInt);
// template void Run<6>(uInt);
// template void Run<7>(uInt);
// template void Run<8>(uInt);
// template void Run<9>(uInt);



int main(int argc, char** argv){
    
    omp_set_num_threads(get_phy_cpu());
    Eigen::setNbThreads(get_phy_cpu());


    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);
                             
    // if(order == 0) Run<0>(meshN);
    if(order == 1) Run<1>(meshN);
    if(order == 2) Run<2>(meshN);
    if(order == 3) Run<3>(meshN);
    if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
    // if(order == 6) Run<6>(meshN);
    // if(order == 7) Run<7>(meshN);
    // if(order == 8) Run<8>(meshN);
    // if(order == 9) Run<9>(meshN);
}