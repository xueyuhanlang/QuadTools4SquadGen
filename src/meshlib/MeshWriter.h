#pragma once

#include "Mesh3D.h"

//////////////////////////////////////////////////////////////////////////
std::array<unsigned char, 3> hsvToRgb(float h, float s, float v);
std::vector<std::array<unsigned char, 3>> generateDistinctColors(int n);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename, const std::vector<ptrdiff_t> *face_cluster_ids = 0, bool use_coloring_algorithm = true);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveChartEdge_as_ply(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename, const std::vector<ptrdiff_t> &face_cluster_ids, float normaloffset = 0.0f);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveMarkedEdge_as_ply(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename, const std::vector<bool> &marked_edge_tag, float normaloffset = 0.0f);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveEdge_as_ply(const std::string &filename, const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::pair<size_t, size_t>> &edges);
template <typename Real>
void SaveEdge_as_ply(const std::string &filename, const std::vector<MeshLib::HE_edge<Real> *> &edges);
//////////////////////////////////////////////////////////////////////////
void SaveMeshlabMLP(const std::string &mlpfile, const std::string &meshfile, const std::string &chartedgefile,
                    int meshedge_width = 1, int charedge_wirewidth = 5);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveComponents(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void save_component_as_ply(const std::vector<MeshLib::HE_face<Real> *> &component, const std::string &filename);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void save_mesh(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, const std::string &filename);
template <typename Real>
void save_mesh(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePtsPLY(const std::string &filename, const std::vector<TinyVector<Real, 3>> &points, const std::vector<TinyVector<Real, 3>> *point_normals = 0, const std::vector<Real> *gray_color = 0, bool use_colormap = true);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void save_as_svg(MeshLib::Mesh3D<Real> *m_pmesh, const std::vector<std::array<unsigned char, 3>> &ply_face_color, const char svgfilename[]);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage_and_gray_color(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename,
                                                   const std::vector<Real> *gray_color = 0, bool use_face_color = true, bool use_colormap = true);

template <typename Real>
void SavePLYmesh_with_float_storage_and_gray_color(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename,
                                                   const std::vector<Real> &gray_color, const Real min_color = 0.0f, const Real max_color = 1.0f, const bool use_face_color = true);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage_and_gray_color(const std::string &filename, const std::vector<TinyVector<Real, 3>> &tri_vertices, const std::vector<ptrdiff_t> &tri_faces, const std::vector<Real> *gray_color = 0, bool use_face_color = true, bool use_colormap = true);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYMesh_with_color(const std::string &filename, const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, const std::vector<size_t> *vertex_cluster_ids = 0, const std::vector<size_t> *face_cluster_ids = 0);
