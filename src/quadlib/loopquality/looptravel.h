#pragma once

#include "Mesh3D.h"

template <typename Real>
bool edge_loop_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_edge<Real> *> &edge_loop, const std::vector<bool> &singular_vertex_tag, std::vector<bool> &complex_edge_tag, bool tag_complex);
template <typename Real>
void edge_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_edge<Real> *> &edge_path, const std::vector<bool> &singular_vertex_tag);
template <typename Real>
void face_loop_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_face<Real> *> &face_loop);
template <typename Real>
void face_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_face<Real> *> &face_path);