#pragma once

#include "Mesh3D.h"

template <typename Real>
void mesh_laplacian_smoothing(MeshLib::Mesh3D<Real> *mesh, int iterations, Real taubin_weight = (Real)1);

template <typename Real>
void mesh_winslow_smoothing(MeshLib::Mesh3D<Real> *mesh, int iterations, Real taubin_weight = (Real)1);

template <typename Real>
void mesh_taubin_smoothing(MeshLib::Mesh3D<Real> *mesh, int iterations, bool feature_aware, Real sharp_angle_in_degree = 30);