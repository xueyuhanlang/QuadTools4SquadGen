#pragma once

#include "Mesh3D.h"

//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool CheckMeshValidity(MeshLib::Mesh3D<Real> *mesh);

template <typename Real>
MeshLib::Mesh3D<Real> *LoopQuadProcessing(MeshLib::Mesh3D<Real> *input_mesh, const Real loop_threshold, bool debug = false, bool subdiv = false, int min_face_num = 16);
