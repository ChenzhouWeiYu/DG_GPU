#include "Type.h"
#include "OptimizedMesh/OptimizedMesh.h"
#include "matrix/matrix.h"
#include "DG_Schemes/DG_Basis.h"
#include "DG_Schemes/Convection.h"
#include "DG_Schemes/ImplicitConvection.h"
#include "Eigen/Sparse"
#include "unsupported/Eigen/IterativeSolvers"

#include "exact.h"

typedef Eigen::SparseMatrix<double,Eigen::RowMajor> EigenSpMat;
typedef Eigen::Triplet<double> Triplet;


// constexpr Scalar param_gamma = 1.4;


// template<typename Type>
// inline Type eval_p(Type rho,Type rhoU,Type rhoV,Type rhoW,Type rhoE){
//     return (param_gamma-1)*(rhoE - 0.5*(rhoU*rhoU + rhoV*rhoV + rhoW*rhoW)/rho);
// }

// template<typename Type>
// inline Type eval_E(Type rho,Type rhoU,Type rhoV,Type rhoW,Type p){
//     return p/rho/(param_gamma-1) + 0.5*(rhoU*rhoU + rhoV*rhoV + rhoW*rhoW)/(rho*rho);
// }
// template<typename Type>
// inline Type eval_rhoE(Type rho,Type rhoU,Type rhoV,Type rhoW,Type p){
//     return p/(param_gamma-1) + 0.5*(rhoU*rhoU + rhoV*rhoV + rhoW*rhoW)/(rho*rho);
// }

// template<typename Type>
// inline Type rho_xyz(Type x, Type y, Type z, Type t){
//     // return x>0.8 ? (y>0.8 ? 1.5 : 0.5323) : (y>0.8 ? 0.5323 : 0.138);
//     if (t>0) 
//         return x<1.0/6.0+(1+20*t)/std::sqrt(3.0) ? 8 : 1.4;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 8 : 1.4;
// }
// template<typename Type>
// inline Type u_xyz(Type x, Type y, Type z, Type t){
//     // return x>0.8 ? 0 : 1.206;
//     if (t>0) 
//         return x<1.0/6.0+(1+20*t)/std::sqrt(3.0) ? 8 : 1.4;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 8.25*std::cos(-M_PI/6.0) : 0;
// }
// template<typename Type>
// inline Type v_xyz(Type x, Type y, Type z, Type t){
//     // return y>0.8 ? 0 : 1.206;
//     if (t>0) 
//         return x<1.0/6.0+(1+20*t)/std::sqrt(3.0) ? 8 : 1.4;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 8.25*std::sin(-M_PI/6.0) : 0;
// }
// template<typename Type>
// inline Type w_xyz(Type x, Type y, Type z, Type t){
//     return 0.0;
// }
// template<typename Type>
// inline Type p_xyz(Type x, Type y, Type z, Type t){
//     // return (param_gamma-1)*rho_xyz(x,y,z,t)*e_xyz(x,y,z);
//     // return x>0.8 ? (y>0.8 ? 1.5 : 0.3) : (y>0.8 ? 0.3 : 0.029); 
//     if (t>0) 
//         return x<1.0/6.0+(1+20*t)/std::sqrt(3.0) ? 8 : 1.4;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 116.5 : 1;
// }
// template<typename Type>
// inline Type e_xyz(Type x, Type y, Type z, Type t){
//     // Scalar r2 = x*x + y*y;
//     // constexpr Scalar r_ds2 = 1.0Q/(2.0Q * 0.1Q * 0.1Q);
//     // return 1e-12 + 0.979264*M_1_PI*r_ds2* std::exp(-r2*r_ds2);
//     Type p = p_xyz(x,y,z,t);
//     Type u = u_xyz(x,y,z,t);
//     Type v = v_xyz(x,y,z,t);
//     Type w = v_xyz(x,y,z,t);
//     return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v+w*w);
    
// }

constexpr double param_gamma = 2.0;

template<typename Type>
inline Type rho_xyz(Type x, Type y, Type /*z*/, Type /*t*/) {
    return static_cast<Type>(1.0); // 密度恒定为 1
}

template<typename Type>
inline Type theta(Type x, Type y) {
    return std::atan2(static_cast<Type>(y), static_cast<Type>(x));
}

template<typename Type>
inline Type r(Type x, Type y) {
    return std::sqrt(x * x + y * y);
}

template<typename Type>
inline Type U(Type x, Type y, Type t) {
    const Type A = 25.0;
    const Type gamma = param_gamma;
    const Type p0_prime = A * gamma * std::pow(1.0, gamma - 1); // 50
    const Type r_val = r<Type>(x, y);
    const Type theta_val = theta<Type>(x, y);

    if (r_val * r_val < 2 * t * t * p0_prime)
        return 0; // 真空区

    const Type sqrt_term = std::sqrt(2 * p0_prime) * std::sqrt(r_val * r_val - 2 * t * t * p0_prime);
    const Type cos_theta = std::cos(theta_val);
    const Type sin_theta = std::sin(theta_val);

    return (2 * t * p0_prime * cos_theta + sqrt_term * sin_theta) / r_val;
}

template<typename Type>
inline Type V(Type x, Type y, Type t) {
    const Type A = 25.0;
    const Type gamma = param_gamma;
    const Type p0_prime = A * gamma * std::pow(1.0, gamma - 1); // 50
    const Type r_val = r<Type>(x, y);
    const Type theta_val = theta<Type>(x, y);

    if (r_val * r_val < 2 * t * t * p0_prime)
        return 0; // 真空区

    const Type sqrt_term = std::sqrt(2 * p0_prime) * std::sqrt(r_val * r_val - 2 * t * t * p0_prime);
    const Type cos_theta = std::cos(theta_val);
    const Type sin_theta = std::sin(theta_val);

    return (2 * t * p0_prime * sin_theta - sqrt_term * cos_theta) / r_val;
}

template<typename Type>
inline Type u_xyz(Type x, Type y, Type z, Type t) {
    return U<Type>(x, y, t);
}

template<typename Type>
inline Type v_xyz(Type x, Type y, Type z, Type t) {
    return V<Type>(x, y, t);
}

template<typename Type>
inline Type w_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>(0.0); // 二维问题，w=0
}

template<typename Type>
inline Type p_xyz(Type x, Type y, Type z, Type t) {
    const Type A = 25.0;
    const Type rho = rho_xyz<Type>(x, y, z, t);
    return A * std::pow(rho, param_gamma);
}

template<typename Type>
inline Type e_xyz(Type x, Type y, Type z, Type t) {
    const Type p = p_xyz<Type>(x, y, z, t);
    const Type rho = rho_xyz<Type>(x, y, z, t);
    const Type u = u_xyz<Type>(x, y, z, t);
    const Type v = v_xyz<Type>(x, y, z, t);
    const Type w = w_xyz<Type>(x, y, z, t);

    return p / (rho * (param_gamma - 1)) + 0.5 * (u * u + v * v + w * w);
}

#define Filed_Func(filedname) \
inline Scalar filedname##_xyz(const vector3f& xyz, Scalar t){\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
inline Scalar filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
inline Scalar rho##filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return rho_xyz(x,y,z,t)*filedname##_xyz(x,y,z,t);\
}
Filed_Func(rho);
Filed_Func(u);
Filed_Func(v);
Filed_Func(w);
Filed_Func(p);
Filed_Func(e);

#undef Filed_Func



template<uInt Order>
void Run(uInt N){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    auto logging = [&](std::string ss){debug("Time  " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    debug("Start   " + std::to_string(chrone_clock()));
    // GeneralMesh mesh = OrthHexMesh({-1.1, -1.1, -1.1/N},{1.1, 1.1, 1.1/N},{N,N,1});
    // GeneralMesh mesh = OrthHexMesh({0.0, 0.0, 0.0},{1.0, 1.0, 1.0/N},{N,N,1});
    // GeneralMesh mesh = OrthHexMesh({0.0, 0.0, 0.0},{3.2, 1.0, 1.0/N},{(18*N)/5,N,1});
    GeneralMesh mesh = OrthHexMesh({-1, -1, 0.0},{1.0, 1.0, 2.0/N},{N,N,1});
    mesh.split_hex5_scan();                                   
    mesh.rebuild_cell_topology();                             
    mesh.validate_mesh();                                     
    ComputingMesh cmesh(mesh);                                
    cmesh.m_boundary.resize(cmesh.m_faces.size());            
    for(uInt fid=0;fid<cmesh.m_faces.size();fid++){           
        if(cmesh.m_faces[fid].m_neighbor_cells[1]==uInt(-1)){ 
            const auto& face = cmesh.m_faces[fid];            
            if(std::abs(face.m_normal[2])>0.5 )                
                cmesh.m_boundary[fid] = 2;
            else{
                const auto& nodes = face.m_nodes;
                const auto& p0 = cmesh.m_points[nodes[0]];
                const auto& p1 = cmesh.m_points[nodes[1]];
                const auto& p2 = cmesh.m_points[nodes[2]];
                const auto& p3 = cmesh.m_points[nodes[3]];
                const auto& centor = (p0+p1+p2+p3)/4;
                // if(centor[0]>1.0/6.0 && centor[1] == 0){
                //     cmesh.m_boundary[fid] = 3;
                // }
                // else
                {
                    cmesh.m_boundary[fid] = 1;
                }
                
            }
            
        }
    }
    logging("Split Hex Mesh to Tet");

    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;

    LongVector<5*Basis::NumBasis> x(mesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi(cell,Xi); });
        const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi);});
        const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi);});
        const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi);});
        const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi);});
        #pragma omp simd
        for(uInt k=0;k<Basis::NumBasis;k++){
            x[cid][5*k+0] = rho_coef[k];
            x[cid][5*k+1] = rhou_coef[k];
            x[cid][5*k+2] = rhov_coef[k];
            x[cid][5*k+3] = rhow_coef[k];
            x[cid][5*k+4] = rhoe_coef[k];
        }
    }
    // print(x);
    
    logging("Set Init Value");


    ExplicitConvection<Basis::OrderBasis> explicitconvection;
    ImplicitConvection<Basis::OrderBasis> convection;
    // LongVector<5*Basis::NumBasis> rhs(x.size());
    // BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
    
    // convection.assemble(cmesh,x,0.0,sparse_mat,rhs);
    // sparse_mat.finalize();


    // const auto& ex_conv = explicitconvection.eval(cmesh, x, 0.0);
    // const auto& im_conv = sparse_mat.multiply(x) - rhs;
    // // print(im_conv);
    // // print(ex_conv);
    // print(max(abs(im_conv - ex_conv)));  //  1e-14

    
    // logging("1111111111");
    
    LongVector<5*Basis::NumBasis> mass(x.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    mass[cid](5*i + k, 0) += phi[i] * phi[i] * 
                        QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                }
            }
        }
    }
    
    // logging("22222222222");
    // print(sparse_mat);
    // sparse_mat.output_as_scalar("./12345.txt");

    // // auto chrono_start = std::chrono::steady_clock::now();
    // // auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    print(std::array<std::string,8>{"#       time", "rel.err  rho",
                    "rel.err  u", "rel.err  v", "rel.err  w", 
                    "rel.err  e", "rel.err coef", "cpu time"});
    for(uInt kkkk=0;kkkk<10;kkkk++){    
        // Scalar dt = 0.001;
        
        // logging("Iter    " + std::to_string(kkkk+1));
        
        // Scalar h = 10.0/N * 0.5773502691896257645091487805019574556476 / 3.0,   vel = 3.12;
        // Scalar max_dt = 0.9 * h * 1.0/(2*Order+1) * 1.0 / vel;
        
        Scalar Dt = 0.0045;


        Scalar max_dt = Dt * std::pow((1.0/N),(Order+1));
        max_dt = Dt * std::pow((1.0/N),Order);

        Scalar dt = max_dt;
        uInt kk = 0;
        
        // debug(dt);
        Scalar sub_t = 0;
        do{ 

            // if(sub_t == 0){
            //     dt = 1e-3 * max_dt;
            // }
            // else{
            //     dt = 1e-2 * max_dt;
            // }
            // dt = 1e-2 * max_dt;
            // if(kkkk>1 || kk>3){
            //     dt = 1.0 * max_dt;
            // }else if(kk==0){
            //     dt = 0.1 * max_dt;
            // }else if(kk==1){
            //     dt = 0.2 * max_dt;
            // }else if(kk==2){
            //     dt = 0.3 * max_dt;
            // }else if(kk==3){
            //     dt = 0.4 * max_dt;
            // }
            // kk++;

            dt = std::min(dt,Dt-sub_t);
            sub_t += dt;
            

            
            LongVector<5*Basis::NumBasis> dx = x;
            // print(dx);

            for(uInt picard_iter = 0; picard_iter < 100; picard_iter++){

                LongVector<5*Basis::NumBasis> rhs(x.size());
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                

                // 用 保存的 dx 代入、离散
                convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);
                sparse_mat.finalize();
                // sparse_mat.output_as_scalar("./12345.txt");

                std::vector<Triplet> tripletList;
                uInt ELL_row_length = sparse_mat.storage.ell_blocks.size();
                uInt ELL_max_row = sparse_mat.storage.ell_max_per_row;
                uInt CSR_nnz = sparse_mat.storage.csr_blocks.size();
                uInt NNZ = (ELL_row_length * ELL_max_row + CSR_nnz) * 5*Basis::NumBasis * 5*Basis::NumBasis;
                uInt ROWs = sparse_mat.num_block_rows * 5*Basis::NumBasis;
                tripletList.reserve(NNZ);

                for(uInt brow=0; brow<sparse_mat.num_block_rows; ++brow){
                    for(uInt i=0; i<sparse_mat.storage.ell_max_per_row; ++i){
                        const uInt bcol = sparse_mat.storage.ell_cols[brow][i];
                        if(bcol == sparse_mat.invalid_index) continue;
                        const auto& block = sparse_mat.storage.ell_blocks[brow][i];
                        for(uInt row=0;row<block.rows();row++){
                            for(uInt col=0;col<block.cols();col++){
                                // if(std::abs(block(row,col))>1e-15)
                                tripletList.push_back(Triplet(brow*block.rows()+row,bcol*block.cols()+col,block(row,col)));
                            }
                        }
                    }

                    const uInt start = sparse_mat.storage.csr_row_ptr[brow];
                    const uInt end = sparse_mat.storage.csr_row_ptr[brow+1];
                    for(uInt idx = start; idx < end; ++idx) {
                        const uInt bcol = sparse_mat.storage.csr_cols[idx];
                        const auto& block = sparse_mat.storage.csr_blocks[idx];

                        for(uInt row=0;row<block.rows();row++){
                            for(uInt col=0;col<block.cols();col++){
                                // if(std::abs(block(row,col))>1e-15)
                                tripletList.push_back(Triplet(brow*block.rows()+row,bcol*block.cols()+col,block(row,col)));
                            }
                        }
                    }
                }
                // for(...)
                // {
                // // ...
                // tripletList.push_back(T(i,j,v_ij));
                // }
                EigenSpMat Eigen_A(ROWs,ROWs);
                // debug(Eigen_A.nonZeros());
                // debug(Eigen_A.rows());
                // debug(Eigen_A.cols());
                // debug(Eigen_A.size());
                Eigen::VectorXd Eigen_b(ROWs), Eigen_x(ROWs);


                Scalar rdt = 1/dt;
                // if(picard_iter < 14){
                //     rdt = 1e1/dt;
                // }
                // if(picard_iter < 12){
                //     rdt = 1e2/dt;
                // }
                // if(picard_iter < 9){
                //     rdt = 1e4/dt;
                // }
                // if(picard_iter < 6){
                //     rdt = 1e8/dt;
                // }
                // if(picard_iter < 3){
                //     rdt = 1e12/dt;
                // }
                // if(picard_iter < 8){
                //     if(picard_iter < 2){
                //         rdt = 1e-2 / dt;
                //     }else if(picard_iter < 4){
                //         rdt = 2e-2 / dt;
                //     }else if(picard_iter < 6){
                //         rdt = 3e-2 / dt;
                //     }else if(picard_iter < 8){
                //         rdt = 4e-2 / dt;
                //     }
                // }else if(picard_iter < 10){
                //     rdt = 2e-1 / dt;
                // }else if(picard_iter < 12){
                //     rdt = 3e-1 / dt;
                // }else if(picard_iter < 14){
                //     rdt = 4e-1 / dt;
                // }
                // 但是 时间 u^n 还是旧的 x
                for(uInt r = 0; r < mass.size(); r++){
                    for(uInt rr = 0; rr < mass[r].size(); rr++){
                        tripletList.push_back(Triplet(r*mass[r].size()+rr, r*mass[r].size()+rr, mass[r][rr]*rdt));
                    }
                }
                rhs += mass*rdt * x;
                // print(dx);



                Eigen_A.setFromTriplets(tripletList.begin(), tripletList.end());
                for(uInt r=0;r<rhs.size();r++){
                    auto& block = rhs[r];
                    for(uInt rr=0;rr<block.size();rr++){
                        Eigen_b[r * block.size() + rr] = block[rr];
                    }
                }

                Eigen::DGMRES<EigenSpMat> Eigen_solver;
                // Eigen::SparseLU<EigenSpMat> Eigen_solver;
                // Eigen_solver.setMaxIterations(200);
                Eigen_solver.setTolerance(1e-12);
                Eigen_solver.compute(Eigen_A);
                if(Eigen_solver.info()!=Eigen::Success) {
                    sparse_mat.output_as_scalar("12345.txt");
                    print(dx);
                    throw std::runtime_error("Matrix decomposition failed");
                }
                Eigen_x = Eigen_solver.solve(Eigen_b);
                // if(Eigen_solver.info()!=Eigen::Success) {
                //     throw std::runtime_error("Matrix solving failed");
                // }

                // 解出来一个新的 dx
                auto ddx = dx;
                Scalar delta = 0.0;
                for(uInt r=0;r<ddx.size();r++){
                    auto& block = ddx[r];
                    for(uInt rr=0;rr<block.size();rr++){
                        Scalar xxx = block[rr] - Eigen_x[r * block.size() + rr];
                        delta += xxx*xxx;
                        block[rr] = Eigen_x[r * block.size() + rr];
                    }
                }
                for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
                    // ddx[cid][2 + 5*0] = 0;
                    ddx[cid][3 + 5*0] = 0;
                    for (uInt k = 1; k < Basis::NumBasis; k++) {
                        // ddx[cid][2 + 5*k] = 0;
                        ddx[cid][3 + 5*k] = 0;
                    }
                }
                

                // 保正限制器
                // 对密度 rho，其平均值为 rho_avg = c_0 * phi_0，而 phi_0 = 1，所以 rho_avg = c_0
                // 考虑密度 rho 在每一个积分点上，都 > 0，或者 > eps = 1e-12, 1e-16 这种
                //   具体来说  对每一个积分点处，计算每一个 rho(x_g)，要求 rho(x_g) > eps
                //   实现方法：找 rho(x_g) 的最小值 rho_min
                //   rho_new = theta * (rho - rho_avg) + rho_avg = theta*rho + (1-theta)*rho_avg
                //       theta = min { 1, (rho_avg-eps)/(rho_avg-rho_min) }
                //   守恒型：rho_new_avg = theta*rho_avg + (1-theta)*rho_avg = rho_avg
                // 尝试改  保极值

                // 先计算一个 上一时刻  单元及其邻居上的  最大最小
                LongVector<5> per_cell_max(cmesh.m_cells.size());
                LongVector<5> per_cell_min(cmesh.m_cells.size());
                for(uInt fid = 0; fid < cmesh.m_faces.size(); fid++){
                    uInt neig_1 = cmesh.m_faces[fid].m_neighbor_cells[0];
                    uInt neig_2 = cmesh.m_faces[fid].m_neighbor_cells[1];
                    // 这里用 dx，是更新前的 U 的 coefficient
                    const auto& coef_1 = dx[neig_1]; 
                    // 判断是否另一单元（边界）
                    const auto& coef_2 = (neig_2==uInt(-1)) ? dx[neig_1] : dx[neig_2]; 
                    Scalar max_scalar = std::numeric_limits<Scalar>::max();
                    DenseMatrix<5,1> max_U = DenseMatrix<5,1>::Ones() * -max_scalar;
                    DenseMatrix<5,1> min_U = DenseMatrix<5,1>::Ones() * max_scalar;
                    for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
                        const auto& xg = QuadC::points[xgi];
                        const auto& basis = Basis::eval_all(xg[0],xg[1],xg[2]);
                        // 计算积分点 xg 处，所有基函数组合的完整的解
                        for (int k = 0; k < 5; ++k) {
                            Scalar val1 = 0.0, val2 = 0.0;
                            for (uInt l = 0; l < Basis::NumBasis; ++l) {
                                val1 += basis[l] * coef_1[k + 5*l];
                                val2 += basis[l] * coef_2[k + 5*l];
                            }
                            // 计算 neig_1,2 单元上，5个守恒量的最大值
                            max_U[k] = std::max(max_U[k], std::max(val1,val2));
                            min_U[k] = std::min(max_U[k], std::min(val1,val2));
                        }
                    }
                    // 将最值 写入到两个单元
                    for (int k = 0; k < 5; ++k) {
                        per_cell_max[neig_1][k] = std::max(per_cell_max[neig_1][k],max_U[k]);
                        per_cell_min[neig_1][k] = std::min(per_cell_min[neig_1][k],min_U[k]);
                        if (neig_2!=uInt(-1)){
                            per_cell_max[neig_2][k] = std::max(per_cell_max[neig_2][k],max_U[k]);
                            per_cell_min[neig_2][k] = std::min(per_cell_min[neig_2][k],min_U[k]);
                        }
                    }
                }

                for (uInt cid = 0; cid < cmesh.m_cells.size(); ++cid) {
                    // 获取当前单元的 max/min 范围
                    auto global_min = per_cell_min[cid];
                    auto global_max = per_cell_max[cid];

                    for (int k = 0; k < 5; ++k) {  // 遍历守恒变量
                        Scalar avg = dx[cid][k];  // 单元平均值

                        // 构造当前变量的积分点值分布
                        std::array<Scalar, QuadC::num_points> qp_vals;
                        qp_vals.fill(0.0);
                        for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
                            const auto& basis = Basis::eval_all(QuadC::points[xgi][0], QuadC::points[xgi][1], QuadC::points[xgi][2]);
                            Scalar val = 0.0;
                            for (uInt l = 0; l < Basis::NumBasis; ++l) {
                                val += basis[l] * dx[cid][k + 5*l];
                            }
                            qp_vals[xgi] = val;
                        }

                        Scalar val_min = *std::min_element(qp_vals.begin(), qp_vals.end());
                        Scalar val_max = *std::max_element(qp_vals.begin(), qp_vals.end());

                        if (val_min < global_min[k] || val_max > global_max[k]) {
                            Scalar theta_min = (avg != val_min) ? (avg - global_min[k]) / (avg - val_min + 1e-32) : 0.0;
                            Scalar theta_max = (avg != val_max) ? (global_max[k] - avg) / (val_max - avg + 1e-32) : 0.0;
                            Scalar theta = std::min({theta_min, theta_max, 1.0});

                            // 应用于高阶模态系数
                            for (uInt l = 1; l < Basis::NumBasis; ++l) {
                                dx[cid][k + 5*l] *= theta;
                            }
                        }
                    }
                }

                #pragma omp parallel for schedule(dynamic)
                for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
                    Scalar eps = 1e-16;
                    std::array<Scalar, QuadC::num_points> rho_qp;
                    rho_qp.fill(0.0);
                    for (uInt xgi = 0; xgi < QuadC::num_points; xgi++) {
                        const auto& xg = QuadC::points[xgi];
                        const auto& basis = Basis::eval_all(xg[0],xg[1],xg[2]);
                        for (uInt k = 0; k < Basis::NumBasis; k++) {
                            rho_qp[xgi] += basis[k] * ddx[cid][0 + 5*k];  // Density
                        }
                    }
                    Scalar rho_min = *std::min_element(rho_qp.begin(), rho_qp.end());
                    Scalar rho_max = *std::max_element(rho_qp.begin(), rho_qp.end());
                    Scalar rho_avg = ddx[cid][0];
                    if (rho_min < eps) {
                        Scalar effective_rho_avg = std::max(rho_avg, eps);
                        Scalar numerator = effective_rho_avg - eps;
                        Scalar denominator = effective_rho_avg - rho_min;

                        Scalar theta = (denominator < 1e-32) ? 0.0 : std::min(1.0, numerator / denominator);

                        // Apply to high-order modes of density
                        for (uInt k = 1; k < Basis::NumBasis; k++) {
                            ddx[cid][0 + 5*k] *= theta;
                        }
                    }
                }

                

                // 定义 lambda 表达式计算压强
                auto compute_pressure = [](const DenseMatrix<5,1>& U) -> Scalar {
                    Scalar rho = U[0];
                    Scalar rhou = U[1];
                    Scalar rhov = U[2];
                    Scalar rhow = U[3];
                    Scalar rhoE = U[4];

                    if (rho <= 1e-16) return -1.0; // 密度为零或负值，压强也应为负

                    Scalar ke = (rhou * rhou + rhov * rhov + rhow * rhow) / (2.0 * rho);
                    return (param_gamma - 1.0) * (rhoE - ke);
                };

                // 对每个单元 cid 进行 压强的 保正
                // #pragma omp parallel for schedule(dynamic)
                for (uInt cid = 0; cid < cmesh.m_cells.size(); ++cid) {
                    // 获取单元平均值
                    DenseMatrix<5,1> avg_U;
                    for (int k = 0; k < 5; ++k) {
                        avg_U[k] = ddx[cid][k]; 
                    }

                    Scalar theta_p = 1.0;

                    // 对每一个积分点
                    for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
                        const auto& basis = Basis::eval_all(QuadC::points[xgi][0], QuadC::points[xgi][1], QuadC::points[xgi][2]);

                        // 计算积分点 xg 处，所有基函数组合的完整的解
                        DenseMatrix<5,1> point_U;
                        for (int k = 0; k < 5; ++k) {
                            Scalar val = 0.0;
                            for (uInt l = 0; l < Basis::NumBasis; ++l) {
                                val += basis[l] * ddx[cid][k + 5*l];
                            }
                            point_U[k] = val;
                        }

                        // 检查当前点压强是否为负，飞醋才需要走下一步
                        Scalar p_gl = compute_pressure(point_U);
                        if (p_gl >= 0) continue;

                        // 二分法
                        Scalar t_low = 0.0, t_high = 1.0;
                        Scalar t_mid = 0.5 * (t_low + t_high);
                        for (int iter = 0; iter < 100; ++iter) {
                            t_mid = 0.5 * (t_low + t_high);
                            DenseMatrix<5,1> U_mid;
                            for (int k = 0; k < 5; ++k) {
                                U_mid[k] = (1.0 - t_mid) * avg_U[k] + t_mid * point_U[k];
                            }

                            Scalar p_mid = compute_pressure(U_mid);

                            if (p_mid < 0) {
                                t_high = t_mid;
                            } else {
                                t_low = t_mid;
                            }
                            
                        }

                        // 全局最小 theta_p
                        // if(cid==0) print(vector2f{theta_p,t_mid});
                        theta_p = std::min(theta_p, t_mid);
                    }
                    // if(cid==0) print(theta_p);
                    for (int k = 0; k < 5; ++k) {          // 遍历每个守恒变量
                        for (uInt l = 1; l < Basis::NumBasis; ++l) {  // 只修改高阶项
                            ddx[cid][k + 5*l] *= theta_p;
                        }
                    }
                }

                delta = (ddx - dx).dot(ddx - dx);
                dx = ddx;



                // const auto& xxxx = sparse_mat.multiply(dx);
                // Scalar delta = xxxx.dot(xxxx);
                delta = std::sqrt(delta);
                std::ostringstream oss;
                oss << delta;
                // std::ostringstream oss1;
                // oss1 << Eigen_solver.error();
                // logging("BiCGSTAB it " + std::to_string(Eigen_solver.iterations()) + "  " + oss1.str());
                {
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    

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
                if( picard_iter > 16 && delta < 1e-10) break;
            }
            // Picard 结束，赋值，x 为 u^{n+1}
            x = dx;// - dt / mass * dx;

            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));


        }while(sub_t < Dt);


        Scalar curr_time = Dt * (kkkk+1);

        std::ofstream fp;
        fp.open("./Order_" + std::to_string(Basis::OrderBasis) + "/Riemann_T_" + std::to_string(kkkk+1) 
                + "_N_" + std::to_string(N) + ".txt");
        auto err_integral = [&](LongVector<QuadC::num_points> uh,LongVector<QuadC::num_points> us){
            Scalar err_per_cells = 0.0, sol_per_cells = 1e-47;
            for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
                const auto& cell = cmesh.m_cells[cid];
                for(uInt g=0; g<QuadC::num_points; ++g) {
                    const auto& weight = QuadC::weights[g] * cell.compute_jacobian_det();
                    Scalar error_cell = uh[cid][g] - us[cid][g];
                    err_per_cells += std::pow(error_cell,2) * weight;
                    sol_per_cells += std::pow(us[cid][g],2) * weight;
                }
            }
            return std::pow(err_per_cells/sol_per_cells,0.5);
        };

        LongVector<QuadC::num_points> rho_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> rho_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> u_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> u_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> v_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> v_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> w_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> w_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> e_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> e_s(cmesh.m_cells.size());
        LongVector<5 * Basis::NumBasis> error(cmesh.m_cells.size());

        #pragma omp parallel for schedule(dynamic)
        for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
            const auto& cell = cmesh.m_cells[cid];
            // rho[cid] = 0.0;
            // auto centor_cid = (cid/5)*5+4;
            // auto centor_cell = cmesh.m_cells[centor_cid];
            for(uInt g=0; g<QuadC::num_points; ++g) {
                auto p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                // Scalar centor = centor_cell.m_centroid[0];
                // Scalar h = 1.0/(QuadC::num_points-1);
                // const auto& pos = vector3f{centor + (g*h-0.5)/N, 0.5/N, 0.5/N};
                // for(centor_cid = (cid/5)*5; centor_cid < (cid/5)*5+5; centor_cid++){
                //     const auto& centor_cell = cmesh.m_cells[centor_cid];
                //     const auto& p0 = cmesh.m_points[centor_cell.m_nodes[0]];
                //     const auto& p1 = cmesh.m_points[centor_cell.m_nodes[0]];
                //     const auto& p2 = cmesh.m_points[centor_cell.m_nodes[0]];
                //     const auto& p3 = cmesh.m_points[centor_cell.m_nodes[0]];
                //     DenseMatrix<3,3> mat({
                //         p1[0]-p0[0],p2[0]-p0[0],p3[0]-p0[0], 
                //         p1[1]-p0[1],p2[1]-p0[1],p3[1]-p0[1], 
                //         p1[2]-p0[2],p2[2]-p0[2],p3[2]-p0[2]
                //     });
                //     p = inverse_3x3(mat).multiply(DenseMatrix<3,1>({pos[0]-p0[0],pos[1]-p0[1],pos[2]-p0[2]}));
                //     if (!(p[0]<0 || p[1]<0 || p[2]<0 || p[0]+p[1]+p[2]>1)) break;
                // }
                const auto& value = Basis::eval_all(p[0],p[1],p[2]);
                for(uInt k=0;k<Basis::NumBasis;k++){
                    rho_h[cid][g] += value[k] * x[cid][5*k+0];
                    u_h[cid][g] += value[k] * x[cid][5*k+1];
                    v_h[cid][g] += value[k] * x[cid][5*k+2];
                    w_h[cid][g] += value[k] * x[cid][5*k+3];
                    e_h[cid][g] += value[k] * x[cid][5*k+4];
                }
                u_h[cid][g] /= rho_h[cid][g];
                v_h[cid][g] /= rho_h[cid][g];
                w_h[cid][g] /= rho_h[cid][g];
                e_h[cid][g] /= rho_h[cid][g];
                rho_s[cid][g] = rho_xyz(pos,curr_time);
                u_s[cid][g] = u_xyz(pos,curr_time);
                v_s[cid][g] = v_xyz(pos,curr_time);
                w_s[cid][g] = w_xyz(pos,curr_time);
                e_s[cid][g] = e_xyz(pos,curr_time);
            }
            const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi( cell,Xi,curr_time);});
            const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi,curr_time);});
            const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi,curr_time);});
            const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi,curr_time);});
            const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi,curr_time);});
            #pragma omp simd
            for(uInt k=0;k<Basis::NumBasis;k++){
                error[cid][5*k+0] = x[cid][5*k+0] - rho_coef[k];
                error[cid][5*k+1] = x[cid][5*k+1] - rhou_coef[k];
                error[cid][5*k+2] = x[cid][5*k+2] - rhov_coef[k];
                error[cid][5*k+3] = x[cid][5*k+3] - rhow_coef[k];
                error[cid][5*k+4] = x[cid][5*k+4] - rhoe_coef[k];
            }
        }

        for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
            const auto& cell = cmesh.m_cells[cid];
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                 << "  " <<std::setprecision(16)<<  rho_h[cid][g] << "  " <<std::setprecision(16)<<  rho_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  u_h[cid][g] << "  " <<std::setprecision(16)<<  u_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  v_h[cid][g] << "  " <<std::setprecision(16)<<  v_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  w_h[cid][g] << "  " <<std::setprecision(16)<<  w_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  e_h[cid][g] << "  " <<std::setprecision(16)<<  e_s[cid][g] << std::endl;
            }
        }
        fp.close();

        print(std::array<Scalar,8>{curr_time, err_integral(rho_h,rho_s), 
                        err_integral(u_h,u_s), err_integral(v_h,v_s), err_integral(w_h,w_s), 
                        err_integral(e_h,e_s), std::sqrt(error.dot(error)/x.dot(x)), chrone_clock()});
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
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
    // if(order == 6) Run<6>(meshN);
    // if(order == 7) Run<7>(meshN);
    // if(order == 8) Run<8>(meshN);
    // if(order == 9) Run<9>(meshN);
}