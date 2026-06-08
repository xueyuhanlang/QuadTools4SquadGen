#pragma once

#include "trimerger.h"

template <typename Real>
class BlossomMerger : public Tri2QuadMerger<Real>
{
public:
    BlossomMerger(MeshLib::Mesh3D<Real> *mesh);
    MeshLib::Mesh3D<Real> *get_merged_mesh();

protected:
    void blossom_merge();

protected:
    bool valid_compute = false;
};
