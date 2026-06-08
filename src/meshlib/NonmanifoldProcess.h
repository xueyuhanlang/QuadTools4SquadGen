#pragma once

#include "TinyVector.h"
#include <vector>

// Hash function for std::pair<size_t, size_t>
namespace std
{
    template <>
    struct hash<std::pair<size_t, size_t>>
    {
        std::size_t operator()(const std::pair<size_t, size_t> &p) const noexcept
        {
            return std::hash<size_t>()(p.first) ^ (std::hash<size_t>()(p.second) << 1);
        }
    };
}
////////////////////////////////////////////////
bool has_nonmanifold_issue(size_t num_vertices, const std::vector<std::vector<size_t>> &face_indices);
////////////////////////////////////////////////
template <typename Real>
bool merge_boundary_vertices(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &fInd, bool &nonmanifold_issue, bool fast_mode, bool skip_degenerate_faces = true);
////////////////////////////////////////////////
template <typename Real>
void manifold_submesh_extraction(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &fInd, const bool input_is_manifold, bool fast_mode);
////////////////////////////////////////////////
template <typename Real>
bool nonmanifold_merge(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, bool fast_mode = true);
////////////////////////////////////////////////
template <typename Real>
size_t label_connected_components(const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, std::vector<size_t> &face_submesh_ids, const size_t start_id);
////////////////////////////////////////////////