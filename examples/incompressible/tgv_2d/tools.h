#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

ComputingMesh create_mesh(uInt N){
    GeneralMesh mesh = OrthHexMesh({0.0, 0.0, 0.0},{1.0, 1.0, 1.0/N},{N,N,1});
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
            else{
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
            }
                
        }
    }
    return cmesh;
}

template<typename QuadC, typename BasisU, typename BasisP>
std::tuple<LongVector<4*QuadC::num_points>, 
            LongVector<4*QuadC::num_points>, 
            LongVector<3*BasisU::NumBasis+BasisP::NumBasis>> 
reconstruct_solution(const ComputingMesh& cmesh, LongVector<3*BasisU::NumBasis+BasisP::NumBasis>& coef, Scalar curr_time){
    LongVector<4*QuadC::num_points> U_h(cmesh.m_cells.size());
    LongVector<4*QuadC::num_points> U_s(cmesh.m_cells.size());
    constexpr uInt DoFs = 3*BasisU::NumBasis+BasisP::NumBasis;
    LongVector<DoFs> error(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        const auto& uvw_func = [&](vector3f Xi)->DenseMatrix<3,1>{
            return uvw_Xi(cell,Xi,curr_time);
        };
        const auto& p_func = [&](vector3f Xi){return p_Xi(cell,Xi,curr_time);};
        const auto& uvw_coef = BasisU::func2coef(uvw_func);
        const auto& p_coef = BasisP::func2coef(p_func);
        for(uInt k=0;k<BasisU::NumBasis;k++){
            const auto& block_coef = coef[cellId].template SubMat<3,1>(3*k,0) - uvw_coef[k];
            MatrixView<DoFs,1,3,1>(error[cellId],3*k,0) = block_coef;
        }
        for(uInt k=0;k<BasisP::NumBasis;k++){
            error[cellId](3*BasisU::NumBasis+k,0) = coef[cellId](3*BasisU::NumBasis+k,0) - p_coef[k];
        }
        for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
            const auto& pos = QuadC::points[xgId];
            // const auto& pos = cell.transform_to_physical(pos);
            const auto& u_value = BasisU::eval_all(pos[0],pos[1],pos[2]);
            const auto& p_value = BasisP::eval_all(pos[0],pos[1],pos[2]);
            const auto& uvw_coef = coef[cellId].template SubMat<3*BasisU::NumBasis,1>(0,0);
            const auto& p_coef = coef[cellId].template SubMat<BasisP::NumBasis,1>(3*BasisU::NumBasis,0);
            const auto& uvw = BasisU::template coef2filed<3,Scalar>(uvw_coef,pos);
            const auto& p = BasisP::template coef2filed<1,Scalar>(p_coef,pos);
            U_h[cellId].template View<3,1>(4*xgId,0) = uvw;
            U_s[cellId].template View<3,1>(4*xgId,0) = uvw_func(pos);
            U_h[cellId](4*xgId+3,0) = p[0];
            U_s[cellId](4*xgId+3,0) = p_func(pos);
        }
    }
    return {U_h, U_s, error};

}