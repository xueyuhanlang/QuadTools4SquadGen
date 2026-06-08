#pragma once

#include "trimerger.h"

// define compare function
template <typename Real>
struct compare_edge
{
	bool operator()(const std::pair<MeshLib::HE_edge<Real> *, Real> &lhs, const std::pair<MeshLib::HE_edge<Real> *, Real> &rhs) const
	{
		return lhs.second > rhs.second;
	}
};

template <typename Real>
class LoopMerger : public Tri2QuadMerger<Real>
{
public:
    LoopMerger(MeshLib::Mesh3D<Real> *mesh, int smooth_num = 5);
    MeshLib::Mesh3D<Real> *get_merged_mesh();

protected:
    void merge_edge(MeshLib::HE_edge<Real> *edge);
    bool opposite_to_singularity(MeshLib::HE_edge<Real> *edge);
    bool connected_with_singularity(MeshLib::HE_edge<Real> *edge);
    void loop_merge();
    bool trigger_degree_2_vertex(MeshLib::HE_edge<Real> *edge);
    bool tri_merge(bool disable_edge_connected_with_singularity);
    bool local_shift(bool disable_edge_opposite_to_singularity);
    void insert_edge_for_merging(const TinyVector<Real, 3> &normal, MeshLib::HE_edge<Real> *dir, const TinyVector<Real, 3> &v01, const TinyVector<Real, 3> &v23, const Real angle_bound = 15);
    bool merge_possibility(MeshLib::HE_edge<Real> *edge, bool dir_switch, const TinyVector<Real, 3> &normal, const TinyVector<Real, 3> &ref_v01, const TinyVector<Real, 3> &ref_v23, Real &angle_difference, Real &regularity);
    Real quadflow_alignment_score(MeshLib::HE_edge<Real> *edge, int *num_neighbor_quads = 0);
    bool loop_shifting();
    int vertex_valid_degree(MeshLib::HE_vert<Real> *vert, bool &boundary_vertex_tag);
    bool valid_for_loop_shifting(MeshLib::HE_edge<Real> *edge, const bool relaxed);
    void vertex_smoothing(int max_iter = 5);
    bool cleanup();
    void corner_case_merge(MeshLib::HE_edge<Real> *edge, MeshLib::HE_edge<Real> *e0, MeshLib::HE_edge<Real> *e1);

protected:
    MeshLib::Mesh3D<Real> *m_pmesh;
    std::vector<TinyVector<Real, 3>> backup_positions;
    std::vector<bool> edge_mergeable_tag;
    std::vector<Real> edge_quad_regularities;
    std::vector<TinyVector<Real, 3>> edge_quad_normals, edge_quad_directions;
    std::priority_queue<std::pair<MeshLib::HE_edge<Real> *, Real>, std::deque<std::pair<MeshLib::HE_edge<Real> *, Real>>, compare_edge<Real>> edge_queue;
    int laplacian_smooth_iter = 5;
    std::vector<TinyVector<Real, 3>> new_positions;
    std::vector<int> vertex_smooth_valence;
};