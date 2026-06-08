#pragma once

#include "Mesh3D.h"
#include "MyTuple.h"

//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real compute_angle(const Real dot_product_value);
template <typename Real>
Real compute_angle(const TinyVector<Real, 3> &unit_dir0, const TinyVector<Real, 3> &unit_dir1);
template <typename Real>
Real compute_angle(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2);
template <typename Real>
Real compute_dihedral_angle(MeshLib::HE_edge<Real> *edge);
template <typename Real>
Real compute_dihedral_angle_diag02(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3);
template <typename Real>
Real quad_dihedral_angle_via_split_quad(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3);

template <typename Real>
Real quad_regularity(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3);

template <typename Real>
bool split_quad(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3);

template <typename Real>
TinyVector<Real, 3> rotate_along_axes(const TinyVector<Real, 3> &V, const TinyVector<Real, 3> &unit_axes, const Real angle);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void compute_statistics(const std::vector<Real> &data, Real &avg_v, Real &max_v, Real &var_v);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void FittingPlane(const std::vector<TinyVector<Real, 3>> &points, const bool closed, TinyVector<Real, 3> &line_direction, TinyVector<Real, 3> &line_origin);
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void get_meshinfo(MeshLib::Mesh3D<Real> *mesh,
				  MySortedTuple<ptrdiff_t, 11, false> &info_tuple,
				  TinyVector<Real, 3> &size, bool apply_transformation = false);
//////////////////////////////////////////////////
template <typename Real>
bool skip_quad_mesh(MeshLib::Mesh3D<Real> *quad_mesh, size_t &num_interior_singularities, size_t &num_boundary_singularities, Real &dangle_diff, int &max_boundary_degree, int &max_interior_degree, int &max_num_of_singular_vertices_in_face);
//////////////////////////////////////////////////
template <typename Real>
bool has_zero_length_edge(MeshLib::Mesh3D<Real> *mesh, const Real tol = (Real)1.0e-10);
//////////////////////////////////////////////////
template <typename Real>
bool quick_filter(MeshLib::Mesh3D<Real> *quad_mesh);