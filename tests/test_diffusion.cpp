#include <fstream>
#include <sstream>
#include <chrono>
#include "Type.h"
#include "OptimizedMesh/OptimizedMesh.h"
#include "matrix/matrix.h"
#include "DG_Schemes/DG_Basis.h"
#include "DG_Schemes/Convection.h"
#include "DG_Schemes/ExplicitDiffusion.h"



constexpr Scalar init_x0 = 3;
constexpr Scalar init_y0 = 3;

constexpr Scalar param_phi = 5;
constexpr Scalar param_gamma = 1.4;
constexpr Scalar param_mu = 1.4;

constexpr Scalar velocity_u = 1.0;
constexpr Scalar velocity_v = 0.5;

template<typename Type>
inline Type rho_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    constexpr Type slope = (param_phi*param_phi*(param_gamma-1))/(16*M_PI*M_PI*param_gamma);
    Type rho = std::pow(1-slope*std::exp(2*(1-r2)), 1.0/(param_gamma-1));
    return 1.0;
}
template<typename Type>
inline Type u_xyz(Type x, Type y, Type z, Type t){
    // Type x0 = init_x0 + velocity_u * t;
    // Type y0 = init_y0 + velocity_v * t;
    // Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    // Type u = velocity_u - param_phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
    Type u = std::sin(2*M_PI*x)*std::cos(2*M_PI*y);
    return u*std::exp(-8*M_PI*M_PI*t*param_mu);
}
template<typename Type>
inline Type v_xyz(Type x, Type y, Type z, Type t){
    // Type x0 = init_x0 + velocity_u * t;
    // Type y0 = init_y0 + velocity_v * t;
    // Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    // Type v = velocity_v + param_phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
    Type v = -std::cos(2*M_PI*x)*std::sin(2*M_PI*y);
    return v*std::exp(-8*M_PI*M_PI*t*param_mu);
}
template<typename Type>
inline Type w_xyz(Type x, Type y, Type z, Type t){
    // Type x0 = init_x0 + velocity_u * t;
    // Type y0 = init_y0 + velocity_v * t;
    // Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    // Type w = 0.0;
    return 0.0;
}
template<typename Type>
inline Type p_xyz(Type x, Type y, Type z, Type t){
    // Type x0 = init_x0 + velocity_u * t;
    // Type y0 = init_y0 + velocity_v * t;
    // Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    // Type p = std::pow(rho_xyz(x,y,z,t),param_gamma);
    Type p = rho_xyz(x,y,x)/4*(std::cos(2*2*M_PI*x)+std::cos(2*2*M_PI*y))*std::exp(-16*M_PI*M_PI*t*param_mu);
    return p;
}
template<typename Type>
inline Type e_xyz(Type x, Type y, Type z, Type t){
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v);
}
template<typename Type>
inline Type fe_xyz(Type x, Type y, Type z, Type t){
    Scalar cos2x std::cos(2*M_PI*x);
    Scalar cos4x std::cos(4*M_PI*x);
    Scalar cos6x std::cos(6*M_PI*x);
    Scalar cos2y std::cos(2*M_PI*y);
    Scalar cos4y std::cos(4*M_PI*y);
    Scalar cos6y std::cos(6*M_PI*y);
    Scalar exp24 = 0.5*std::exp(-24*M_PI*M_PI*t*param_mu)*M_PI;
    Scalar CosCos62 = cos6x*cos2y-cos2x*cos6y;
    Scalar exp16 = 4.0/(param_gamma-1)*std::exp(-16*M_PI*M_PI*t*param_mu)*M_PI*M_PI*param_mu;
    Scalar CosCos4 = cos4x+cos4y+2*(1-param_gamma)*cos4x*cos4y;
    return exp24*CosCos62+exp16*CosCos4;
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
Filed_Func(fe);












int main(int argc, char** argv){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    
    omp_set_num_threads(get_phy_cpu());
    uInt N=20;
    GeneralMesh mesh = OrthHexMesh({0,0,0},{1,1,1.0/N},{N,N,1});
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


    // LongVector<QuadCC::num_points> rho_h(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> rho_s(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> u_h(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> u_s(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> v_h(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> v_s(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> w_h(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> w_s(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> e_h(cmesh.m_cells.size());
    // LongVector<QuadCC::num_points> e_s(cmesh.m_cells.size());
    // LongVector<5 * Basis::NumBasis> error(cmesh.m_cells.size());

    // #pragma omp parallel for schedule(dynamic)
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     const auto& cell = cmesh.m_cells[cid];
    //     // rho[cid] = 0.0;
    //     for(uInt g=0; g<QuadCC::num_points; ++g) {
    //         const auto& p = QuadCC::points[g];
    //         const auto& pos = cell.transform_to_physical(p);
    //         const auto& value = Basis::eval_all(p[0],p[1],p[2]);
    //         for(uInt k=0;k<Basis::NumBasis;k++){
    //             rho_h[cid][g] += value[k] * x[cid][5*k+0];
    //             u_h[cid][g] += value[k] * x[cid][5*k+1];
    //             v_h[cid][g] += value[k] * x[cid][5*k+2];
    //             w_h[cid][g] += value[k] * x[cid][5*k+3];
    //             e_h[cid][g] += value[k] * x[cid][5*k+4];
    //         }
    //         u_h[cid][g] /= rho_h[cid][g];
    //         v_h[cid][g] /= rho_h[cid][g];
    //         w_h[cid][g] /= rho_h[cid][g];
    //         e_h[cid][g] /= rho_h[cid][g];
    //         rho_s[cid][g] = rho_xyz(pos,0.0);
    //         u_s[cid][g] = u_xyz(pos,0.0);
    //         v_s[cid][g] = v_xyz(pos,0.0);
    //         w_s[cid][g] = w_xyz(pos,0.0);
    //         e_s[cid][g] = e_xyz(pos,0.0);
    //     }
    // }

    // debug(std::array<Scalar,5>{ err_integral(rho_h,rho_s), 
    //             err_integral(u_h,u_s), err_integral(v_h,v_s), err_integral(w_h,w_s), 
    //             err_integral(e_h,e_s)});


    // ExplicitDiffusion<Basis::OrderBasis> diffusion;
    
    // LongVector<5*Basis::NumBasis> b(x.size());
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
    //     for(uInt g=0; g<QuadC::num_points; ++g) {
    //         const auto& p = QuadC::points[g];
    //         auto phi = Basis::eval_all(p[0], p[1], p[2]);
    //         for(uInt i=0; i<Basis::NumBasis; ++i) {
    //             for(uInt k=0; k<5; ++k) {
    //                 b[cid](5*i + k, 0) += phi[i] * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
    //             }
    //         }
    //     }
    // }


    // Scalar dt = 1e-4;
    // const auto& k1 = - 1/b * diffusion.eval(cmesh, x);
    // const auto& k2 = - 1/b * diffusion.eval(cmesh, x + 1.0/2.0 * dt * k1);
    // x = x + dt * k2;
    // // debug(dt*k1);
    // // debug(dt*k2);
    // // debug(x);

    //     std::ofstream fp;
    //     fp.open("./Order_diffusion_3");
    //     fp <<std::setprecision(16)<< ("#       x") << "  " 
    //         <<std::setprecision(16)<< (" y") << "  " 
    //         <<std::setprecision(16)<< (" z")
    //              << "  " <<std::setprecision(16)<<  (" rho")
    //              << "  " <<std::setprecision(16)<<  (" u")
    //              << "  " <<std::setprecision(16)<<  (" v")
    //              << "  " <<std::setprecision(16)<<  (" w")
    //              << "  " <<std::setprecision(16)<<  (" e") << std::endl;

    //     LongVector<QuadC::num_points> rho_h(cmesh.m_cells.size());
    //     LongVector<QuadC::num_points> u_h(cmesh.m_cells.size());
    //     LongVector<QuadC::num_points> v_h(cmesh.m_cells.size());
    //     LongVector<QuadC::num_points> w_h(cmesh.m_cells.size());
    //     LongVector<QuadC::num_points> e_h(cmesh.m_cells.size());
    //     // LongVector<5 * Basis::NumBasis> error(cmesh.m_cells.size());

    //     #pragma omp parallel for schedule(dynamic)
    //     for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //         const auto& cell = cmesh.m_cells[cid];
    //         // rho[cid] = 0.0;
    //         for(uInt g=0; g<QuadC::num_points; ++g) {
    //             const auto& p = QuadC::points[g];
    //             const auto& pos = cell.transform_to_physical(p);
    //             const auto& value = Basis::eval_all(p[0],p[1],p[2]);
    //             for(uInt k=0;k<Basis::NumBasis;k++){
    //                 rho_h[cid][g] += value[k] * x[cid][5*k+0];
    //                 u_h[cid][g] += value[k] * x[cid][5*k+1];
    //                 v_h[cid][g] += value[k] * x[cid][5*k+2];
    //                 w_h[cid][g] += value[k] * x[cid][5*k+3];
    //                 e_h[cid][g] += value[k] * x[cid][5*k+4];
    //             }
    //             u_h[cid][g] /= rho_h[cid][g];
    //             v_h[cid][g] /= rho_h[cid][g];
    //             w_h[cid][g] /= rho_h[cid][g];
    //             e_h[cid][g] /= rho_h[cid][g];
    //         }
    //     }

    //     for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //         const auto& cell = cmesh.m_cells[cid];
    //         for(uInt g=0; g<QuadC::num_points; ++g) {
    //             const auto& p = QuadC::points[g];
    //             const auto& pos = cell.transform_to_physical(p);
    //             fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
    //              << "  " <<std::setprecision(16)<<  rho_h[cid][g]
    //              << "  " <<std::setprecision(16)<<  u_h[cid][g] 
    //              << "  " <<std::setprecision(16)<<  v_h[cid][g] 
    //              << "  " <<std::setprecision(16)<<  w_h[cid][g]
    //              << "  " <<std::setprecision(16)<<  e_h[cid][g] << std::endl;
    //         }
    //     }
    //     fp.close();




// //========================= 测隐格式的 =========================//

    ExplicitConvection<Basis::OrderBasis> convection;
    ExplicitDiffusion<Basis::OrderBasis> diffusion;
    
    LongVector<5*Basis::NumBasis> b(x.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    b[cid](5*i + k, 0) += phi[i] * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                }
            }
        }
    }

    LongVector<5*Basis::NumBasis> f(x.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            auto fe = fe_Xi(cmesh.m_cells[cid],p,0.0);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                // for(uInt k=0; k<5; ++k) {
                f[cid](5*i + 4, 0) += fe * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                // }
            }
        }
    }

    for(uInt kkkk=0;kkkk<40;kkkk++){
        // const auto& y = 1/b * convection.eval(cmesh, x);
        Scalar dt = 0.001;
        Scalar h = 10.0/N * std::pow(1.0/3.0,0.5),   vel = 3.12;
        // debug(h / (2*5+1) / vel);
        #pragma unroll
        for(uInt sub_step = 0;sub_step < 100; sub_step++){
            const auto& k1 = 1/b * (diffusion.eval(cmesh, x) - convection.eval(cmesh, x));
            const auto& k2 = 1/b * (diffusion.eval(cmesh, x + 1.0/2.0 * dt * k1) - convection.eval(cmesh, x + 1.0/2.0 * dt * k1));
            // const auto& k3 = - 1/b * convection.eval(cmesh, x + 1.0/2.0 * dt * k2);
            // const auto& k4 = - 1/b * convection.eval(cmesh, x + 1.0/1.0 * dt * k3);

            // x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
            x = x + dt * k2;// (k1 + k2) / 2.0;
            // debug(sub_step);
        }
        Scalar curr_time = dt * 100 * (kkkk+1);
        // debug(0.01 * y);
        // debug(kkkk);


        // fp.open("./Order_" + std::to_string(Basis::OrderBasis) + "/rho_T_" + std::to_string(kkkk+1) 
        //         + "_NT_" + std::to_string((uInt)(1.0/(0.0005*10)))  
        //         + "_N_" + std::to_string(N) + "_HighOrder.txt");
        LongVector<QuadC::num_points> rho(cmesh.m_cells.size());
        LongVector<QuadC::num_points> err(cmesh.m_cells.size());

        for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
            const auto& cell = cmesh.m_cells[cid];
            rho[cid] = 0.0;
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                const auto& value = Basis::eval_all(p[0],p[1],p[2]);
                for(uInt k=0;k<Basis::NumBasis;k++){
                    rho[cid][g] += value[k] * x[cid][5*k+0];
                }
                err[cid][g] = rho[cid][g] - rho_xyz(pos,curr_time);
                // fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
            }
        }
        debug(vector3f{curr_time, std::sqrt(err.dot(err)/rho.dot(rho)),chrone_clock()});
        // fp.close();
    }

    
}