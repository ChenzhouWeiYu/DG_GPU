#pragma once
#include "mesh/general_mesh.h"

class OrthHexMesh: public GeneralMesh{

public:
    OrthHexMesh() = default;
    OrthHexMesh(vector3f lb, vector3f ub, vector3u Nxyz){
        uInt Nx = Nxyz[0];
        uInt Ny = Nxyz[1];
        uInt Nz = Nxyz[2];
        m_points.resize((Nx+1)*(Ny+1)*(Nz+1));
        m_cells.resize((Nx)*(Ny)*(Nz));
        for(auto&c:m_cells) c=Hexahedron();
        m_faces.resize((Nx)*(Ny)*(Nz+1) + (Nx+1)*(Ny)*(Nz) + (Nx)*(Ny+1)*(Nz));

        for(uInt i=0;i<Nx+1;i++){
            for(uInt j=0;j<Ny+1;j++){
                for(uInt k=0;k<Nz+1;k++){
                    uInt idx = i + (j + k*(Ny+1)) * (Nx+1);
                    m_points[idx] = { lb[0] + i * (ub[0]-lb[0])/Nx,
                                      lb[1] + j * (ub[1]-lb[1])/Ny,
                                      lb[2] + k * (ub[2]-lb[2])/Nz};
                }
            }
        }

        for(uInt i=0;i<Nx;i++){
            for(uInt j=0;j<Ny;j++){
                for(uInt k=0;k<Nz+1;k++){
                    uInt idx = i + (j + k*(Ny)) * (Nx) + 0;
                    if(k==0 || k==Nz){
                        uInt p0 = (i+0) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p1 = (i+1) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p2 = (i+1) + ((j+1) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p3 = (i+0) + ((j+1) + (k+0)*(Ny+1)) * (Nx+1);
                        vector4u face2node = {p0,p1,p2,p3};
                        vector2u face2cell = {i + (j + (k==Nz?k-1:k)*(Ny)) * (Nx), uInt(-1)};
                        m_faces[idx] = QuadFace{face2node,face2cell};
                    }
                    else{
                        uInt p0 = (i+0) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p1 = (i+1) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p2 = (i+1) + ((j+1) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p3 = (i+0) + ((j+1) + (k+0)*(Ny+1)) * (Nx+1);
                        vector4u face2node = {p0,p1,p2,p3};
                        vector2u face2cell = {i + (j + (k-1)*(Ny)) * (Nx), i + (j + k*(Ny)) * (Nx)};
                        m_faces[idx] = QuadFace{face2node,face2cell};
                    }
                }
            }
        }

        for(uInt i=0;i<Nx+1;i++){
            for(uInt j=0;j<Ny;j++){
                for(uInt k=0;k<Nz;k++){
                    uInt idx = i + (j + k*(Ny)) * (Nx+1) + (Nx)*(Ny)*(Nz+1);
                    if(i==0 || i==Nx){
                        uInt p0 = (i+0) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p1 = (i+0) + ((j+1) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p2 = (i+0) + ((j+1) + (k+1)*(Ny+1)) * (Nx+1);
                        uInt p3 = (i+0) + ((j+0) + (k+1)*(Ny+1)) * (Nx+1);
                        vector4u face2node = {p0,p1,p2,p3};
                        vector2u face2cell = {(i==Nx?i-1:i) + (j + k*(Ny)) * (Nx), uInt(-1)};
                        m_faces[idx] = QuadFace{face2node,face2cell};
                    }
                    else{
                        uInt p0 = (i+0) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p1 = (i+0) + ((j+1) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p2 = (i+0) + ((j+1) + (k+1)*(Ny+1)) * (Nx+1);
                        uInt p3 = (i+0) + ((j+0) + (k+1)*(Ny+1)) * (Nx+1);
                        vector4u face2node = {p0,p1,p2,p3};
                        vector2u face2cell = {(i-1) + (j + k*(Ny)) * (Nx), i + (j + k*(Ny)) * (Nx)};
                        m_faces[idx] = QuadFace{face2node,face2cell};
                    }
                }
            }
        }

        for(uInt i=0;i<Nx;i++){
            for(uInt j=0;j<Ny+1;j++){
                for(uInt k=0;k<Nz;k++){
                    uInt idx = i + (j + k*(Ny+1)) * (Nx) + (Nx)*(Ny)*(Nz+1) + (Nx+1)*(Ny)*(Nz);
                    if(j==0 || j==Ny){
                        uInt p0 = (i+0) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p1 = (i+0) + ((j+0) + (k+1)*(Ny+1)) * (Nx+1);
                        uInt p2 = (i+1) + ((j+0) + (k+1)*(Ny+1)) * (Nx+1);
                        uInt p3 = (i+1) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        vector4u face2node = {p0,p1,p2,p3};
                        vector2u face2cell = {i + ((j==Ny?j-1:j) + k*(Ny)) * (Nx), uInt(-1)};
                        m_faces[idx] = QuadFace{face2node,face2cell};
                    }
                    else{
                        uInt p0 = (i+0) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        uInt p1 = (i+0) + ((j+0) + (k+1)*(Ny+1)) * (Nx+1);
                        uInt p2 = (i+1) + ((j+0) + (k+1)*(Ny+1)) * (Nx+1);
                        uInt p3 = (i+1) + ((j+0) + (k+0)*(Ny+1)) * (Nx+1);
                        vector4u face2node = {p0,p1,p2,p3};
                        vector2u face2cell = {i + ((j-1) + k*(Ny)) * (Nx), i + (j + k*(Ny)) * (Nx)};
                        m_faces[idx] = QuadFace{face2node,face2cell};
                    }
                }
            }
        }
        rebuild_cell_topology();
        validate_mesh();
    }
};