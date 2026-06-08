#pragma once

#include "TinyVector.h"
#include <vector>
#include <string>
#include "Mesh3D.h"
////////////////////////////////////////////////
template <typename Real>
void mesh_load_interface(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, bool nonmanifold_input = false);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_stl(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_ply(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_off(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_obj(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_glb(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_vtk(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_vtp(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_wrl(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
void mesh_load_fbx(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *create_mesh(std::vector<MeshLib::HE_face<Real> *> component);
template <typename Real>
MeshLib::Mesh3D<Real> *create_mesh(const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, bool ignore_nonmanifold = true);
template <typename Real>
void mesh_decomposition(MeshLib::Mesh3D<Real> *mesh, std::vector<MeshLib::Mesh3D<Real> *> &submeshes);
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *load_mesh(const std::string &filename, bool nonmanifold_input = false, bool ignore_nonmanifold = true);
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *LoadPLYmesh(const std::string &filename, std::vector<std::array<unsigned char, 3>> &ply_face_color);
////////////////////////////////////////////////
template <typename Real>
void mesh_to_vertices_and_faces(MeshLib::Mesh3D<Real> *m_pmesh, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////