#include <fstream>
#include <sstream>
#include <chrono>
#include "Type.h"
#include "OptimizedMesh/OptimizedMesh.h"
#include "matrix/matrix.h"
#include "DG_Schemes/DG_Basis.h"
#include "DG_Schemes/Convection.h"

#include "exact.h"


constexpr Scalar init_x0 = 4.0;
constexpr Scalar init_y0 = 4.5;

constexpr Scalar param_phi = 5;
constexpr Scalar param_gamma = 1.4;

constexpr Scalar velocity_u = 1.0;
constexpr Scalar velocity_v = 0.5;


template<typename Type>
inline Type rho_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    constexpr Type slope = (param_phi*param_phi*(param_gamma-1))/(16*M_PI*M_PI*param_gamma);
    Type rho = std::pow(1-slope*std::exp(2*(1-r2)), 1.0/(param_gamma-1));
    return rho;
}
template<typename Type>
inline Type u_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type u = velocity_u - param_phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
    return u;
}
template<typename Type>
inline Type v_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type v = velocity_v + param_phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
    return v;
}
template<typename Type>
inline Type w_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type w = 0.0;
    return w;
}
template<typename Type>
inline Type p_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type p = std::pow(rho_xyz(x,y,z,t),param_gamma);
    return p;
}
template<typename Type>
inline Type e_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v);
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
    GeneralMesh mesh = OrthHexMesh({0,0,0},{10,10,10.0/N},{N,N,1});
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
    
    logging("Set Init Value");


    ExplicitConvection<Basis::OrderBasis> convection;
    
    LongVector<5*Basis::NumBasis> b(x.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    b[cid](5*i + k, 0) += phi[i] * phi[i] * 
                        QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                }
            }
        }
    }

    // auto chrono_start = std::chrono::steady_clock::now();
    // auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    print(std::array<std::string,8>{"#       time", "rel.err  rho",
                    "rel.err  u", "rel.err  v", "rel.err  w", 
                    "rel.err  e", "rel.err coef", "cpu time"});
    for(uInt kkkk=0;kkkk<10;kkkk++){    
        // Scalar dt = 0.001;
        
        // logging("Iter    " + std::to_string(kkkk+1));
        
        Scalar h = 10.0/N * 0.5773502691896257645091487805019574556476 / 3.0,   vel = 3.12;
        Scalar max_dt = 0.9 * h * 1.0/(2*Order+1) * 1.0 / vel;
        Scalar dt = max_dt;
        // debug(dt);
        Scalar sub_t = 0;
        do{
            dt = std::min(max_dt,0.1-sub_t);
            sub_t += dt;

            // debug(vector2f{dt,sub_t});
            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));
            const auto& k1 = - 1/b * convection.eval(cmesh, x, 
                                                            0.1 * (kkkk) + sub_t);
            const auto& x1 = x + 1.0/2.0 * dt * k1;
            // for(uInt cid=0;cid<x1.size();cid++){
            //     x1[cid][5*0+0] = std::max(1e-4,x1[cid][5*0+0]);
            // }
            const auto& k2 = - 1/b * convection.eval(cmesh, x1,  
                                                            0.1 * (kkkk) + sub_t + 1.0/2.0 * dt); 
            x = x + dt * k2;
            // for(uInt cid=0;cid<x1.size();cid++){
            //     x[cid][5*0+0] = std::max(1e-4,x[cid][5*0+0]);
            // }

            // const auto& h1 = - 1/b * convection.eval(cmesh, x);
            // const auto& h2 = - 1/b * convection.eval(cmesh, x + 0.5 * dt * h1);
            // const auto& h3 = - 1/b * convection.eval(cmesh, x + 0.5 * dt * h2);
            // const auto& h4 = - 1/b * convection.eval(cmesh, x + 1.0 * dt * h3);
            // x = x + dt * (h1 + 2*h2 + 2*h3 + h4) / 6;

        }while(sub_t < 0.1);


        Scalar curr_time = 0.1 * (kkkk+1);

        std::ofstream fp;
        fp.open("./Order_" + std::to_string(Basis::OrderBasis) + "/rho_T_" + std::to_string(kkkk+1) 
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
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
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

template void Run<0>(uInt);
template void Run<1>(uInt);
template void Run<2>(uInt);
template void Run<3>(uInt);
template void Run<4>(uInt);
template void Run<5>(uInt);
template void Run<6>(uInt);
template void Run<7>(uInt);
template void Run<8>(uInt);
template void Run<9>(uInt);



int main(int argc, char** argv){
    

    omp_set_num_threads(get_phy_cpu());


    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);
                             
    if(order == 0) Run<0>(meshN);
    if(order == 1) Run<1>(meshN);
    if(order == 2) Run<2>(meshN);
    if(order == 3) Run<3>(meshN);
    if(order == 4) Run<4>(meshN);
    if(order == 5) Run<5>(meshN);
    if(order == 6) Run<6>(meshN);
    if(order == 7) Run<7>(meshN);
    if(order == 8) Run<8>(meshN);
    if(order == 9) Run<9>(meshN);
}