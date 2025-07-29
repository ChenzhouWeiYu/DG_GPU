#include <fstream>
#include <sstream>
#include <chrono>
#include "Type.h"
#include "OptimizedMesh/OptimizedMesh.h"
#include "matrix/matrix.h"
#include "DG_Schemes/DG_Basis.h"
#include "DG_Schemes/Convection.h"



constexpr Scalar init_x0 = 3;
constexpr Scalar init_y0 = 3;

constexpr Scalar param_phi = 5;
constexpr Scalar param_gamma = 1.4;

constexpr Scalar velocity_u = 1.0;
constexpr Scalar velocity_v = 0.5;

template<typename Type>
inline Type rho_xyz(Type x, Type y, Type z, Type t = 0.0){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    constexpr Type slope = (param_phi*param_phi*(param_gamma-1))/(16*M_PI*M_PI*param_gamma);
    Type rho = std::pow(1-slope*std::exp(2*(1-r2)), 1.0/(param_gamma-1));
    return rho;
}
template<typename Type>
inline Type u_xyz(Type x, Type y, Type z, Type t = 0.0){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type u = velocity_u - param_phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
    return u;
}
template<typename Type>
inline Type v_xyz(Type x, Type y, Type z, Type t = 0.0){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type v = velocity_v + param_phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
    return v;
}
template<typename Type>
inline Type w_xyz(Type x, Type y, Type z, Type t = 0.0){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type w = 0.0;
    return w;
}
template<typename Type>
inline Type p_xyz(Type x, Type y, Type z, Type t = 0.0){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type p = std::pow(rho_xyz(x,y,z,t),param_gamma);
    return p;
}
template<typename Type>
inline Type e_xyz(Type x, Type y, Type z, Type t = 0.0){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v);
}

#define Filed_Func(filedname) \
inline Scalar filedname##_xyz(const vector3f& xyz, Scalar t = 0.0){\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
inline Scalar filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t = 0.0){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
inline Scalar rho##filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t = 0.0){\
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












int main(int argc, char** argv){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    
    omp_set_num_threads(get_phy_cpu());
    uInt N=20;
    GeneralMesh mesh = OrthHexMesh({0,0,0},{10,10,10.0/N},{N,N,1});
    mesh.split_hex6_scan();
    mesh.rebuild_cell_topology();
    mesh.validate_mesh();
    ComputingMesh cmesh(mesh);
    cmesh.m_boundary.resize(cmesh.m_faces.size());
    for(uInt fid=0;fid<cmesh.m_faces.size();fid++){
        if(cmesh.m_faces[fid].m_neighbor_cells[1]==uInt(-1)){
            const auto& face = cmesh.m_faces[fid];
            if(std::abs(face.m_normal[2])>0.5)
                cmesh.m_boundary[fid] = 2;
            else
                cmesh.m_boundary[fid] = 1;
        }
    }

    // debug("Mesh  " + std::to_string(chrone_clock()));

    // for(uInt fid=0;fid<cmesh.m_faces.size();fid++){
    //     if(cmesh.m_faces[fid].m_neighbor_cells[1]==uInt(-1)||cmesh.m_faces[fid].m_neighbor_cells[0]==uInt(-1)){
    //        debug(cmesh.m_faces[fid].m_neighbor_cells);
    //     }
    // }

    
    // 会牵扯到积分点选取，影响到代数精度的，还有：
    //   - DGBasisEvaluator 里面 的 最后一段，给定函数（物理场）最小二乘确定系数
    //   - Convection 里面 的 两个积分
    //   - 质量矩阵，用的也是下面这个 QuadC
    // 下面的 QuadC 的积分点，还被用于计算误差、重构密度场的坐标点选取（不影响精度，只是在这些点上计算）
    
    using Basis = DGBasisEvaluator<4>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;



    // constexpr uInt P=3, NumPoly=(P+1)*(P+2)*(P+3)/6;
    LongVector<5*Basis::NumBasis> x(mesh.m_cells.size());

    // 计算所有基函数在给定点的值
    auto values = Basis::eval_all(0.5, 0.5, 0.0);

    // 计算所有基函数在给定点的梯度
    auto grads = Basis::grad_all(0.5, 0.5, 0.0);

    // 访问第5个基函数的值和梯度
    Scalar val_p5 = values[4]; 
    auto grad_p5 = grads[4];

    // debug(val_p5);
    // debug(grad_p5);
    // for(uInt i=0;i<x.size();i++){
    //     x[i][0]=1.0;
    //     x[i][1]=8.41;
    //     x[i][2]=3.22;
    //     x[i][3]=0.0;
    //     x[i][4]=1.78;
    // }
    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi(cell,Xi); });
        const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi);});
        const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi);});
        const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi);});
        const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi);});

        // const long double xc = cell.m_centroid[0], yc = cell.m_centroid[1], zc = cell.m_centroid[2];
        // const long double err = e_xyz(xc,yc,zc) - e_xyz(cell.m_centroid[0],cell.m_centroid[1],cell.m_centroid[2]);
        // if(err*err > 1e-30){
        //     debug(err);
        // }

        #pragma omp simd
        for(uInt k=0;k<Basis::NumBasis;k++){
            x[cid][5*k+0] = rho_coef[k];
            x[cid][5*k+1] = rhou_coef[k];
            x[cid][5*k+2] = rhov_coef[k];
            x[cid][5*k+3] = rhow_coef[k];
            x[cid][5*k+4] = rhoe_coef[k];
        }
    }
    // debug("init   " + std::to_string(chrone_clock()));


        using QuadCC = GaussLegendreTet::Degree11Points94;

    auto err_integral = [&](LongVector<QuadCC::num_points> uh,LongVector<QuadCC::num_points> us){
        Scalar err_per_cells = 0.0, sol_per_cells = 1e-47;
        for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
            const auto& cell = cmesh.m_cells[cid];
            for(uInt g=0; g<QuadCC::num_points; ++g) {
                const auto& weight = QuadCC::weights[g] * cell.compute_jacobian_det();
                Scalar error_cell = uh[cid][g] - us[cid][g];
                err_per_cells += std::pow(error_cell,2) * weight;
                sol_per_cells += std::pow(us[cid][g],2) * weight;
            }
        }
        return std::pow(err_per_cells/sol_per_cells,0.5);
    };


    LongVector<QuadCC::num_points> rho_h(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> rho_s(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> u_h(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> u_s(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> v_h(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> v_s(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> w_h(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> w_s(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> e_h(cmesh.m_cells.size());
    LongVector<QuadCC::num_points> e_s(cmesh.m_cells.size());
    LongVector<5 * Basis::NumBasis> error(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        // rho[cid] = 0.0;
        for(uInt g=0; g<QuadCC::num_points; ++g) {
            const auto& p = QuadCC::points[g];
            const auto& pos = cell.transform_to_physical(p);
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
            rho_s[cid][g] = rho_xyz(pos,0.0);
            u_s[cid][g] = u_xyz(pos,0.0);
            v_s[cid][g] = v_xyz(pos,0.0);
            w_s[cid][g] = w_xyz(pos,0.0);
            e_s[cid][g] = e_xyz(pos,0.0);
        }
    }

    debug(std::array<Scalar,5>{ err_integral(rho_h,rho_s), 
                err_integral(u_h,u_s), err_integral(v_h,v_s), err_integral(w_h,w_s), 
                err_integral(e_h,e_s)});

    // LongVector<QuadCC::num_points> rho(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> err(cmesh.m_cells.size());
    
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     const auto& cell = cmesh.m_cells[cid];
    //     rho[cid] = 0.0;
    //     for(uInt g=0; g<QuadCC::num_points; ++g) {
    //         const auto& p = QuadCC::points[g];
    //         const auto& pos = cell.transform_to_physical(p);
    //         const auto& value = Basis::eval_all(p[0],p[1],p[2]);
    //         for(uInt k=0;k<Basis::NumBasis;k++){
    //             rho[cid][g] += value[k] * x[cid][5*k+0];
    //         }
    //         err[cid][g] = rho[cid][g] - rho_xyz(pos);
    //     }
    //     // const auto& single_err = std::sqrt(err[cid].dot(err[cid])/rho[cid].dot(rho[cid]));
    //     // if(single_err>1e-6){
    //     //     debug(cell.m_centroid);
    //     //     debug(single_err);
    //     // }
    // }
    // // debug(std::sqrt(err.dot(err)/rho.dot(rho)));
    // // debug(std::sqrt(err.dot(err)));






    // debug(x);

    // std::ofstream fp;

    // fp.open("rho0.txt");
    // LongVector<QuadC::num_points> rho(cmesh.m_cells.size());
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     const auto& cell = cmesh.m_cells[cid];
    //     rho[cid] = 0.0;
    //     for(uInt g=0; g<QuadC::num_points; ++g) {
    //         const auto& p = QuadC::points[g];
    //         const auto& pos = cell.transform_to_physical(p);
    //         const auto& value = Basis::eval_all(p[0],p[1],p[2]);
    //         for(uInt k=0;k<Basis::NumBasis;k++){
    //             rho[cid][g] += value[k]*x[cid][5*k+0];
    //         }
    //         fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
    //     }
    // }
    // fp.close();

    // debug(x);





//========================= 测隐格式的 =========================//

    // ImplicitConvection<Basis::OrderBasis> convection;
    // BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
    // convection.assemble(cmesh,x,sparse_mat);

    // LongVector<5*Basis::NumBasis> b(x.size());
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
    //     for(uInt g=0; g<QuadC::num_points; ++g) {
    //         const auto& p = QuadC::points[g];
    //         auto phi = Basis::eval_all(p[0], p[1], p[2]);
    //         for(uInt i=0; i<Basis::NumBasis; ++i) {
    //             for(uInt k=0; k<5; ++k) {
    //                 mass_matrix(5*i + k, 5*i + k) += phi[i] * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].m_volume*6;
    //             }
    //         }
    //     }
    //     // debug("000");
    //     const auto& y  = (mass_matrix*1.0/1e-2).multiply(x[cid]);
    //     // debug("2222");
    //     b[cid] = y;
    //     // debug("333");

    //     // sparse_mat.add_block(cid,cid,mass_matrix*1.0/1e-2);
    // }
    
    // // debug("111111111111111111");
    // sparse_mat.finalize();
    // // debug("11111222222222222222");


    // debug(sparse_mat.storage.ell_blocks.size());
    // debug(sparse_mat.storage.ell_cols.size());
    // debug(sparse_mat.storage.ell_max_per_row);
    // debug(sparse_mat.storage.csr_blocks.size());
    // debug(sparse_mat.storage.csr_cols.size());
    // debug(sparse_mat.storage.csr_row_ptr.size());
    // debug(sparse_mat.num_block_rows);
    // std::ofstream fp;
    // // fp.open("A.txt");
    // // // sparse_mat.output_as_scalar("A.txt");
    // // fp<<sparse_mat;
    // // fp.close();

    // // fp.open("b.txt");
    // // sparse_mat.output_as_scalar("A.txt");
    // // fp<<b;
    // // fp.close();
    // // debug(sparse_mat);


    // fp.open("rho.txt");
    // LongVector<QuadC::num_points> rho(cmesh.m_cells.size());
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     const auto& cell = cmesh.m_cells[cid];
    //     rho[cid] = 0.0;
    //     for(uInt g=0; g<QuadC::num_points; ++g) {
    //         const auto& p = QuadC::points[g];
    //         const auto& pos = cell.transform_to_physical(p);
    //         const auto& value = Basis::eval_all(p[0],p[1],p[2]);
    //         for(uInt k=0;k<Basis::NumBasis;k++){
    //             rho[cid][g] += value[k]*x[cid][5*k+0];
    //         }
    //         fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
    //     }
    // }
    
    // // sparse_mat.output_as_scalar("A.txt");
    // // fp<<rho;
    // fp.close();



    // for(uInt kkkk=0;kkkk<5;kkkk++){


        
    //     x = x + 1e-1*sparse_mat.multiply(x);

    //     fp.open("rho" + std::to_string(kkkk+1) + ".txt");
    //     LongVector<QuadC::num_points> rho(cmesh.m_cells.size());
    //     for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //         const auto& cell = cmesh.m_cells[cid];
    //         rho[cid] = 0.0;
    //         for(uInt g=0; g<QuadC::num_points; ++g) {
    //             const auto& p = QuadC::points[g];
    //             const auto& pos = cell.transform_to_physical(p);
    //             const auto& value = Basis::eval_all(p[0],p[1],p[2]);
    //             for(uInt k=0;k<Basis::NumBasis;k++){
    //                 rho[cid][g] += value[k]*x[cid][5*k+0];
    //             }
    //             fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
    //         }
    //     }
        
    //     // sparse_mat.output_as_scalar("A.txt");
    //     // fp<<rho;
    //     fp.close();
    // }








// //========================= 测隐格式的 =========================//

//     ExplicitConvection<Basis::OrderBasis> convection;
    
//     LongVector<5*Basis::NumBasis> b(x.size());
//     for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
//         DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
//         for(uInt g=0; g<QuadC::num_points; ++g) {
//             const auto& p = QuadC::points[g];
//             auto phi = Basis::eval_all(p[0], p[1], p[2]);
//             for(uInt i=0; i<Basis::NumBasis; ++i) {
//                 for(uInt k=0; k<5; ++k) {
//                     b[cid](5*i + k, 0) += phi[i] * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
//                 }
//             }
//         }
//     }

//     // 2860 2864

//     // const auto& cell = std::get<Tetrahedron>(mesh.m_cells[2860]);
//     // debug(cell.faces);
//     // for(const auto& fid:cell.faces){
//     //     const auto& face = std::get<TriangleFace>(mesh.m_faces[fid]);
//     //     debug(face.neighbor_cells);
//     //     debug(face.nodes);
//     //     debug(std::get<Tetrahedron>(mesh.m_cells[face.neighbor_cells[0]]).nodes);
//     //     if(face.neighbor_cells[1]!=uInt(-1))
//     //     debug(std::get<Tetrahedron>(mesh.m_cells[face.neighbor_cells[1]]).nodes);

//     // }

//     // debug(1/b);
//     auto chrono_start = std::chrono::steady_clock::now();
//     auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    
//     for(uInt kkkk=0;kkkk<40;kkkk++){
//         // const auto& y = 1/b * convection.eval(cmesh, x);
//         Scalar dt = 0.001;
//         Scalar h = 10.0/N * std::pow(1.0/3.0,0.5),   vel = 3.12;
//         // debug(h / (2*5+1) / vel);
//         #pragma unroll
//         for(uInt sub_step = 0;sub_step < 100; sub_step++){
//             const auto& k1 = - 1/b * convection.eval(cmesh, x);
//             const auto& k2 = - 1/b * convection.eval(cmesh, x + 1.0/2.0 * dt * k1);
//             // const auto& k3 = - 1/b * convection.eval(cmesh, x + 1.0/2.0 * dt * k2);
//             // const auto& k4 = - 1/b * convection.eval(cmesh, x + 1.0/1.0 * dt * k3);

//             // x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
//             x = x + dt * k2;// (k1 + k2) / 2.0;
//             // debug(sub_step);
//         }
//         Scalar curr_time = dt * 100 * (kkkk+1);
//         // debug(0.01 * y);
//         // debug(kkkk);


//         // fp.open("./Order_" + std::to_string(Basis::OrderBasis) + "/rho_T_" + std::to_string(kkkk+1) 
//         //         + "_NT_" + std::to_string((uInt)(1.0/(0.0005*10)))  
//         //         + "_N_" + std::to_string(N) + "_HighOrder.txt");
//         LongVector<QuadC::num_points> rho(cmesh.m_cells.size());
//         LongVector<QuadC::num_points> err(cmesh.m_cells.size());

//         for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
//             const auto& cell = cmesh.m_cells[cid];
//             rho[cid] = 0.0;
//             for(uInt g=0; g<QuadC::num_points; ++g) {
//                 const auto& p = QuadC::points[g];
//                 const auto& pos = cell.transform_to_physical(p);
//                 const auto& value = Basis::eval_all(p[0],p[1],p[2]);
//                 for(uInt k=0;k<Basis::NumBasis;k++){
//                     rho[cid][g] += value[k] * x[cid][5*k+0];
//                 }
//                 err[cid][g] = rho[cid][g] - rho_xyz(pos,curr_time);
//                 // fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
//             }
//         }
//         debug(vector3f{curr_time, std::sqrt(err.dot(err)/rho.dot(rho)),chrone_clock()});
//         // fp.close();
//     }

    
}