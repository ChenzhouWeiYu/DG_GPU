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

inline Scalar rho_xyz(Scalar x, Scalar y, Scalar z, Scalar t = 0.0){
    Scalar x0 = init_x0 + velocity_u * t;
    Scalar y0 = init_y0 + velocity_v * t;
    Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    constexpr Scalar slope = (param_phi*param_phi*(param_gamma-1))/(16*M_PI*M_PI*param_gamma);
    Scalar rho = std::pow(1-slope*std::exp(2*(1-r2)), 1.0/(param_gamma-1));
    return rho;
}

inline Scalar u_xyz(Scalar x, Scalar y, Scalar z, Scalar t = 0.0){
    Scalar x0 = init_x0 + velocity_u * t;
    Scalar y0 = init_y0 + velocity_v * t;
    Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Scalar u = velocity_u - param_phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
    return u;
}

inline Scalar v_xyz(Scalar x, Scalar y, Scalar z, Scalar t = 0.0){
    Scalar x0 = init_x0 + velocity_u * t;
    Scalar y0 = init_y0 + velocity_v * t;
    Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Scalar v = velocity_v + param_phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
    return v;
}

inline Scalar w_xyz(Scalar x, Scalar y, Scalar z, Scalar t = 0.0){
    Scalar x0 = init_x0 + velocity_u * t;
    Scalar y0 = init_y0 + velocity_v * t;
    Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Scalar w = 0.0;
    return w;
}

inline Scalar p_xyz(Scalar x, Scalar y, Scalar z, Scalar t = 0.0){
    Scalar x0 = init_x0 + velocity_u * t;
    Scalar y0 = init_y0 + velocity_v * t;
    Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Scalar p = std::pow(rho_xyz(x,y,z,t),param_gamma);
    return p;
}
inline Scalar e_xyz(Scalar x, Scalar y, Scalar z, Scalar t = 0.0){
    Scalar x0 = init_x0 + velocity_u * t;
    Scalar y0 = init_y0 + velocity_v * t;
    Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Scalar p = p_xyz(x,y,z,t);
    Scalar u = u_xyz(x,y,z,t);
    Scalar v = v_xyz(x,y,z,t);
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
    uInt N=20;
    GeneralMesh mesh = OrthHexMesh({0,0,0},{10,10,0.1},{N,N,1});
    mesh.split_hex5_scan();
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

    // for(uInt fid=0;fid<cmesh.m_faces.size();fid++){
    //     if(cmesh.m_faces[fid].m_neighbor_cells[1]==uInt(-1)||cmesh.m_faces[fid].m_neighbor_cells[0]==uInt(-1)){
    //        debug(cmesh.m_faces[fid].m_neighbor_cells);
    //     }
    // }

    
    
    using Basis = DGBasisEvaluator<1>;
    using QuadC = GaussLegendreTet::Degree5Points15;



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

    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        auto rho_func = [&](vector3f Xi){
            auto xyz = cell.transform_to_physical(Xi);
            Scalar x=xyz[0],y=xyz[1],z=xyz[2], x0=3,y0=3;
            Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
            Scalar phi=5.0, gamma=1.4;
            Scalar rho = std::pow(1-phi*phi*(gamma-1)/(16*M_PI*M_PI*gamma)*std::exp(2*(1-r2)),1.0/(gamma-1));
            return rho;
        };
        auto rhou_func = [&](vector3f Xi){
            auto xyz = cell.transform_to_physical(Xi);
            Scalar x=xyz[0],y=xyz[1],z=xyz[2], x0=3,y0=3;
            Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
            Scalar phi=5.0, gamma=1.4;
            Scalar rho = std::pow(1-phi*phi*(gamma-1)/(16*M_PI*M_PI*gamma)*std::exp(2*(1-r2)),1.0/(gamma-1));
            Scalar u = 1.0 - phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
            return rho*u;
        };
        auto rhov_func = [&](vector3f Xi){
            auto xyz = cell.transform_to_physical(Xi);
            Scalar x=xyz[0],y=xyz[1],z=xyz[2], x0=3,y0=3;
            Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
            Scalar phi=5.0, gamma=1.4;
            Scalar rho = std::pow(1-phi*phi*(gamma-1)/(16*M_PI*M_PI*gamma)*std::exp(2*(1-r2)),1.0/(gamma-1));
            Scalar v = 0.5 + phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
            return rho*v;
        };
        auto rhow_func = [&](vector3f Xi){
            auto xyz = cell.transform_to_physical(Xi);
            Scalar x=xyz[0],y=xyz[1],z=xyz[2], x0=3,y0=3;
            Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
            Scalar phi=5.0, gamma=1.4;
            Scalar rho = std::pow(1-phi*phi*(gamma-1)/(16*M_PI*M_PI*gamma)*std::exp(2*(1-r2)),1.0/(gamma-1));
            Scalar w = 0.0;
            return rho*w;
        };
        auto rhoe_func = [&](vector3f Xi){
            auto xyz = cell.transform_to_physical(Xi);
            Scalar x=xyz[0],y=xyz[1],z=xyz[2], x0=3,y0=3;
            Scalar r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
            Scalar phi=5.0, gamma=1.4;
            Scalar rho = std::pow(1-phi*phi*(gamma-1)/(16*M_PI*M_PI*gamma)*std::exp(2*(1-r2)),1.0/(gamma-1));
            Scalar p = std::pow(rho,gamma);
            Scalar u = 1.0 - phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
            Scalar v = 0.5 + phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
            return p/(gamma-1)+0.5*rho*(u*u+v*v);
        };
        const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi(cell,Xi); });
        const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi);});
        const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi);});
        const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi);});
        const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi);});
        for(uInt k=0;k<Basis::NumBasis;k++){
            x[cid][5*k+0] = rho_coef[k];
            x[cid][5*k+1] = rhou_coef[k];
            x[cid][5*k+2] = rhov_coef[k];
            x[cid][5*k+3] = rhow_coef[k];
            x[cid][5*k+4] = rhoe_coef[k];
        }
    }
    // debug(x);

    std::ofstream fp;

    fp.open("rho0.txt");
    LongVector<QuadC::num_points> rho(cmesh.m_cells.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        rho[cid] = 0.0;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            const auto& pos = cell.transform_to_physical(p);
            const auto& value = Basis::eval_all(p[0],p[1],p[2]);
            for(uInt k=0;k<Basis::NumBasis;k++){
                rho[cid][g] += value[k]*x[cid][5*k+0];
            }
            fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
        }
    }
    fp.close();

    // debug(x);



//========================= RK4 显格式 =========================//

    ExplicitConvection<Basis::OrderBasis> convection;
    
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

    // debug(1/b);
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    for(uInt kkkk=0;kkkk<800;kkkk++){
        Scalar delta_t = 0.0005;
        for(uInt sub_dt = 0; sub_dt<10; sub_dt++){
            const auto& h1 = - 1/b * convection.eval(cmesh, x);
            const auto& h2 = - 1/b * convection.eval(cmesh, x + 0.5 * delta_t * h1);
            const auto& h3 = - 1/b * convection.eval(cmesh, x + 0.5 * delta_t * h2);
            const auto& h4 = - 1/b * convection.eval(cmesh, x + 1.0 * delta_t * h3);
            x = x + delta_t * (h1 + 2*h2 + 2*h3 + h4) / 6;

        }
        Scalar curr_time = delta_t * 10 * (kkkk+1);


        fp.open("./RK4_Order_" + std::to_string(Basis::OrderBasis) + "/rho_T_" + std::to_string(kkkk+1) 
                + "_NT_" + std::to_string((uInt)(1.0/(0.0005*10)))  
                + "_N_" + std::to_string(N) + ".txt");
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
                fp << pos[0] << "  " << pos[1] << "  " << pos[2] << "  " << rho[cid][g] << std::endl;
            }
        }
        debug(vector3f{curr_time, std::sqrt(err.dot(err)/rho.dot(rho)),chrone_clock()});
        fp.close();
    }

    
}