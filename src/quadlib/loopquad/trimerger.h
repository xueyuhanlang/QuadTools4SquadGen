#pragma once

#include "Mesh3D.h"

template <typename Real>
class Tri2QuadMerger
{
public:
    Tri2QuadMerger(MeshLib::Mesh3D<Real> *mesh);
protected:
    MeshLib::Mesh3D<Real> *create_merged_mesh();
    bool edge_mergeable(MeshLib::HE_edge<Real> *edge, bool check_face_tag = true);
    bool quad_mergeable(MeshLib::HE_edge<Real> *edge, Real &angle_regularity,
                        TinyVector<Real, 3> &normal, TinyVector<Real, 3> &dir0, TinyVector<Real, 3> &dir1, TinyVector<Real, 3> &dir2, TinyVector<Real, 3> &dir3,
                        bool check_face_tag = true);

protected:
    MeshLib::Mesh3D<Real> *m_pmesh;    //!< pointer to the mesh
    Real angle_bound_for_merging = 10; // in degree
    Real flow_alignment_bound = 15;    // in degree
    Real diheral_angle_bound = 120;    // in degree
    Real quad_min_angle_bound = 10;    // in degree
};
