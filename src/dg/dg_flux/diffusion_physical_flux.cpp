#include "dg/dg_flux/diffusion_physical_flux.h"



template<uInt GammaNumerator, uInt GammaDenominator>
Scalar DiffusionPhysicalFlux<GammaNumerator, GammaDenominator>::computePressure(const DenseMatrix<5,1>& U) {
    const Scalar rho = positive(U[0]);
    const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
    return (gamma-1)*(rho*e - 0.5*rho*(u*u + v*v + w*w));
}

template<uInt GammaNumerator, uInt GammaDenominator>
Scalar DiffusionPhysicalFlux<GammaNumerator, GammaDenominator>::positive(Scalar val) { 
    return std::max(val, epslion); 
}


template<uInt GammaNumerator, uInt GammaDenominator>
DenseMatrix<5,3> DiffusionPhysicalFlux<GammaNumerator, GammaDenominator>::
computeFlux(const DenseMatrix<5,1>& U, const DenseMatrix<5,3>& grad_U,const Scalar mu){
    // U = {rho, rho u, rho v, rho w, rho e}
    const Scalar rho = positive(U[0]);
    const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
    const Scalar u2 = u*u + v*v + w*w;
    const Scalar p = computePressure(U);
    const Scalar Const = gamma/Pr;  
    // 对于标量场  s，考虑 rho*s 的梯度 grad_U 和 s 的梯度
    // grad(rho s) = grad(rho) s + rho grad(s)
    // grad(s) = 1/rho * (grad(rho s) - grad(rho) s)
    DenseMatrix<3,3> grad_uvw;
    #pragma GCC unroll 16
    for(uInt i=0;i<3;i++){
        for(uInt j=0;j<3;j++){
            grad_uvw(i,j) = 1/rho*(grad_U(i+1,j)-grad_U(0,j)*U(i+1,0)/rho);
        }
    }
    const DenseMatrix<3,3>& tau_ij = mu*(grad_uvw + grad_uvw.transpose() 
                        - 2.0/3.0*grad_uvw.trace()*DenseMatrix<3,3>::Identity());
    // q = -mu/(Ma^2*Pr*(gamma-1))*grad(T),   p=rho T/(gamma*Ma^2) = rho(gamma-1)(e-0.5(u^2+v^2+w^2))
    // q = -mu/(Pr*(gamma-1))*grad(p*gamma/rho)
    // q = -mu*gamma/(Pr*(gamma-1))*grad((gamma-1)(e-0.5(u^2+v^2+w^2)))
    // q = -mu*gamma/(Pr)*grad(e-0.5(u^2+v^2+w^2))
    // Scalar T = (gamma-1)*(e - 0.5*(u*u + v*v + w*w));
    DenseMatrix<3,1> q = {
        -mu * Const * (1/rho*(grad_U(4,0)-grad_U(0,0)*U(4,0)/rho) - u*grad_uvw(0,0) - v*grad_uvw(1,0) - w*grad_uvw(2,0)),
        -mu * Const * (1/rho*(grad_U(4,1)-grad_U(0,1)*U(4,0)/rho) - u*grad_uvw(0,1) - v*grad_uvw(1,1) - w*grad_uvw(2,1)),
        -mu * Const * (1/rho*(grad_U(4,2)-grad_U(0,2)*U(4,0)/rho) - u*grad_uvw(0,2) - v*grad_uvw(1,2) - w*grad_uvw(2,2))
    };
    return {0,0,0,
            tau_ij(0,0),tau_ij(0,1),tau_ij(0,2),
            tau_ij(1,0),tau_ij(1,1),tau_ij(1,2),
            tau_ij(2,0),tau_ij(2,1),tau_ij(2,2),
            u*tau_ij(0,0)+v*tau_ij(1,0)+w*tau_ij(2,0)-q[0],
            u*tau_ij(0,1)+v*tau_ij(1,1)+w*tau_ij(2,1)-q[1],
            u*tau_ij(0,2)+v*tau_ij(1,2)+w*tau_ij(2,2)-q[2]};
}


template<uInt GammaNumerator, uInt GammaDenominator>
std::array<std::array<DenseMatrix<5,5>,3>,3> DiffusionPhysicalFlux<GammaNumerator, GammaDenominator>::
computeJacobian(const DenseMatrix<5,1>& U,const Scalar mu) {
    const Scalar rho = positive(U[0]);
    const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
    const Scalar u2 = u*u + v*v + w*w;
    const Scalar p = computePressure(U);
    const Scalar Const = gamma/Pr;  
    const auto& G_fx_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -4.0/3.0*u, 4.0/3.0, 0, 0, 0, -v, 0, 1, 0, 0, -w, 0, 0, 1, 0, -Const*e + std::pow(u, 2)*(Const - 4.0/3.0) + (Const - 1)*(std::pow(v, 2) + std::pow(w, 2)), u*(4.0/3.0 - Const), v*(1 - Const), w*(1 - Const), Const});
    const auto& G_fx_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -u, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, v, -2.0/3.0*u, 0, 0});
    const auto& G_fx_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, 0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -1.0/3.0*u*w, w, 0, -2.0/3.0*u, 0});
    const auto& G_fy_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -v, 0, 1, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, -2.0/3.0*v, u, 0, 0});
    const auto& G_fy_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -4.0/3.0*v, 0, 4.0/3.0, 0, 0, -w, 0, 0, 1, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - 4.0/3.0*std::pow(v, 2) - std::pow(w, 2), u*(1 - Const), v*(4.0/3.0 - Const), w*(1 - Const), Const});
    const auto& G_fy_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, -v, 0, 1, 0, 0, -1.0/3.0*v*w, 0, w, -2.0/3.0*v, 0});
    const auto& G_fz_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -w, 0, 0, 1, 0, 0, 0, 0, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, -1.0/3.0*u*w, -2.0/3.0*w, 0, u, 0});
    const auto& G_fz_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -w, 0, 0, 1, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -1.0/3.0*v*w, 0, -2.0/3.0*w, v, 0});
    const auto& G_fz_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -v, 0, 1, 0, 0, -4.0/3.0*w, 0, 0, 4.0/3.0, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - std::pow(v, 2) - 4.0/3.0*std::pow(w, 2), u*(1 - Const), v*(1 - Const), w*(4.0/3.0 - Const), Const});
    return std::array<std::array<DenseMatrix<5,5>,3>,3>{
        {{G_fx_ux, G_fx_uy, G_fx_uz},  {G_fy_ux, G_fy_uy, G_fy_uz},  {G_fz_ux, G_fz_uy, G_fz_uz}}};
}

template<uInt GammaNumerator, uInt GammaDenominator>
DenseMatrix<5,5> DiffusionPhysicalFlux<GammaNumerator, GammaDenominator>::
computevKu(const DenseMatrix<3,1>& left_vec,const DenseMatrix<3,1>& right_vec,const DenseMatrix<5,1>& U,const Scalar mu){
    // const std::array<std::array<DenseMatrix<5, 5>, 3>, 3>& J_G_U = computeJacobian(U,mu);  
    const Scalar rho = positive(U[0]);
    const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
    const Scalar u2 = u*u + v*v + w*w;
    const Scalar p = computePressure(U);
    const Scalar Const = gamma/Pr;  
    const auto& G_fx_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -4.0/3.0*u, 4.0/3.0, 0, 0, 0, -v, 0, 1, 0, 0, -w, 0, 0, 1, 0, -Const*e + std::pow(u, 2)*(Const - 4.0/3.0) + (Const - 1)*(std::pow(v, 2) + std::pow(w, 2)), u*(4.0/3.0 - Const), v*(1 - Const), w*(1 - Const), Const});
    const auto& G_fx_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -u, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, v, -2.0/3.0*u, 0, 0});
    const auto& G_fx_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, 0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -1.0/3.0*u*w, w, 0, -2.0/3.0*u, 0});
    const auto& G_fy_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -v, 0, 1, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, -2.0/3.0*v, u, 0, 0});
    const auto& G_fy_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -4.0/3.0*v, 0, 4.0/3.0, 0, 0, -w, 0, 0, 1, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - 4.0/3.0*std::pow(v, 2) - std::pow(w, 2), u*(1 - Const), v*(4.0/3.0 - Const), w*(1 - Const), Const});
    const auto& G_fy_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, -v, 0, 1, 0, 0, -1.0/3.0*v*w, 0, w, -2.0/3.0*v, 0});
    const auto& G_fz_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -w, 0, 0, 1, 0, 0, 0, 0, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, -1.0/3.0*u*w, -2.0/3.0*w, 0, u, 0});
    const auto& G_fz_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -w, 0, 0, 1, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -1.0/3.0*v*w, 0, -2.0/3.0*w, v, 0});
    const auto& G_fz_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -v, 0, 1, 0, 0, -4.0/3.0*w, 0, 0, 4.0/3.0, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - std::pow(v, 2) - 4.0/3.0*std::pow(w, 2), u*(1 - Const), v*(1 - Const), w*(4.0/3.0 - Const), Const});
    // DenseMatrix<5,5> flux = DenseMatrix<5,5>::Zeros(); 
    // #pragma GCC unroll 16
    // for(uInt mm=0;mm<3;mm++){
    //     #pragma GCC unroll 16
    //     for(uInt nn=0;nn<3;nn++){
    //         flux += v(nn,0) * J_G_U[nn][mm] * u(mm,0);
    //     }
    // }
    // return flux;
    return left_vec[0]*G_fx_ux*right_vec[0] + left_vec[1]*G_fy_ux*right_vec[0] + left_vec[2]*G_fz_ux*right_vec[0] +
           left_vec[0]*G_fx_uy*right_vec[1] + left_vec[1]*G_fy_uy*right_vec[1] + left_vec[2]*G_fz_uy*right_vec[1] +
           left_vec[0]*G_fx_uz*right_vec[2] + left_vec[1]*G_fy_uz*right_vec[2] + left_vec[2]*G_fz_uz*right_vec[2];
}

template<uInt GammaNumerator, uInt GammaDenominator>
DenseMatrix<5,5> DiffusionPhysicalFlux<GammaNumerator, GammaDenominator>::
computevKu(const DenseMatrix<3,1>& left_vec,const DenseMatrix<3,1>& right_vec,const std::array<std::array<DenseMatrix<5, 5>, 3>, 3>& J_G_U){
    // DenseMatrix<5,5> flux = DenseMatrix<5,5>::Zeros(); 
    // #pragma GCC unroll 16
    // for(uInt mm=0;mm<3;mm++){
    //     #pragma GCC unroll 16
    //     for(uInt nn=0;nn<3;nn++){
    //         flux += v(nn,0) * J_G_U[nn][mm] * u(mm,0);
    //     }
    // }
    // return flux;
    const auto& [G_fx_ux, G_fx_uy, G_fx_uz] = J_G_U[0];
    const auto& G_fx_right_vec = G_fx_ux*right_vec[0] + G_fx_uy*right_vec[1] + G_fx_uz*right_vec[2];
    const auto& [G_fy_ux, G_fy_uy, G_fy_uz] = J_G_U[1];
    const auto& G_fy_right_vec = G_fy_ux*right_vec[0] + G_fy_uy*right_vec[1] + G_fy_uz*right_vec[2];
    const auto& [G_fz_ux, G_fz_uy, G_fz_uz] = J_G_U[2];
    const auto& G_fz_right_vec = G_fz_ux*right_vec[0] + G_fz_uy*right_vec[1] + G_fz_uz*right_vec[2];
    return left_vec[0]*G_fx_right_vec + left_vec[1]*G_fy_right_vec + left_vec[2]*G_fz_right_vec;
}













template class DiffusionPhysicalFlux<5,3>;  // gamma=5/3
template class DiffusionPhysicalFlux<7,5>;  // gamma=7/5