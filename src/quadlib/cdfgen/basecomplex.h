#pragma once

#include "Mesh3D.h"
#include <unordered_map>

class complex_arc
{
public:
    complex_arc(const ptrdiff_t vstart_id = -1, const ptrdiff_t vend_id = -1, const ptrdiff_t cluster_id0 = -1, const ptrdiff_t cluster_id1 = -1);
    bool operator==(const complex_arc &c) const;
    bool operator!=(const complex_arc &c) const;
    bool operator<(const complex_arc &c) const;

public:
    ptrdiff_t end_vertices[2];
    ptrdiff_t cluster_id[2];
};

class complex_arc_info
{
public:
    complex_arc_info();
    bool add_ring_neighbor(const ptrdiff_t v1, const ptrdiff_t v2, const ptrdiff_t cluster_id0, const ptrdiff_t cluster_id1);

public:
    // neighbors in the two opposoite arcs
    ptrdiff_t ring_neighbor_arc_1[2], ring_neighbor_arc_2[2];
    ptrdiff_t cluster_id_1[2], cluster_id_2[2];
    int group_id;
    double arclength;
    bool visited;
    bool orientation; // for grouping arcs;
};

namespace std
{
    template <>
    struct hash<complex_arc>
    {
        std::size_t operator()(const complex_arc &obj) const
        {
            std::size_t h1 = std::hash<ptrdiff_t>()(obj.end_vertices[0]);
            std::size_t h2 = std::hash<ptrdiff_t>()(obj.end_vertices[1]);
            std::size_t h3 = std::hash<ptrdiff_t>()(obj.cluster_id[0]);
            std::size_t h4 = std::hash<ptrdiff_t>()(obj.cluster_id[1]);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };
}

template <typename Real>
class BaseComplex
{
public:
    BaseComplex(MeshLib::Mesh3D<Real> *input_mesh, Real sharp_angle = 150);
    ///////////////////////////////////////////////
    const std::vector<bool> &get_boundary_vertex_tag() const;
    ///////////////////////////////////////////////
    const std::vector<bool> &get_singular_vertex_tag() const;
    ///////////////////////////////////////////////
    const std::vector<bool> &get_complex_edge_tag() const;
    /////////////////////////////////////////////////
    const std::vector<bool> &get_corner_tag() const;
    /////////////////////////////////////////////////
    const std::vector<ptrdiff_t> &get_face_patch_ids() const;
    /////////////////////////////////////////////////
    const std::vector<std::vector<MeshLib::HE_edge<Real> *>> &get_complex_edge_loops() const;
    /////////////////////////////////////////////////
    const std::vector<std::vector<ptrdiff_t>> &get_complex_edge_loops_corner_starting_edges() const;
    /////////////////////////////////////////////////
    const std::vector<std::vector<ptrdiff_t>> &get_complex_edge_loops_neighbor_cluster_ids() const;
    /////////////////////////////////////////////////
    const std::vector<ptrdiff_t> &get_complex_edge_loops_cluster_ids() const;
    /////////////////////////////////////////////////
    const std::unordered_map<complex_arc, complex_arc_info> &get_complex_arcs() const;
    const size_t get_arc_group_num() const;
    /////////////////////////////////////////////////
    void export_complex_as_ply(const char filename[], bool save_curved_edges = false);
    /////////////////////////////////////////////////
    const int get_num_singularity() const;
    const int get_num_complex() const;
    /////////////////////////////////////////////////
protected:
    void extract_base_complex();
    ////////////////////////////////////////////////
    void pre_computation();
    ////////////////////////////////////////////////
    void compute_base_patch();
    ////////////////////////////////////////////////
    Real get_edgeloop_length(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop);
    ////////////////////////////////////////////////
    int count_complex_vertices(const std::vector<MeshLib::HE_edge<Real> *> &close_edge_loop, std::vector<int> &count_corners);
    //////////////////////////////////////////////
    void split_edge_loop(std::vector<std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real>> &edge_loops_with_length, bool has_singularity);
    //////////////////////////////////////////////
    void loop_travel_for_split(MeshLib::HE_vert<Real> *vert, std::vector<MeshLib::HE_edge<Real> *> &edge_loop);
    void tag_edge_loop_for_split(std::vector<MeshLib::HE_edge<Real> *> &edge_loop, bool status);
    int find_mid_point(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, Real arclength, int start_pos, int end_pos);
    int find_farest_point(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, Real looplength, int start_pos);
    void find_two_farest_points(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, Real looplength, int start_pos, int &pos0, int &pos1);
    ////////////////////////////////////////////////
    bool is_complex_corner(MeshLib::HE_vert<Real> *hv, bool singular_tag);
    //////////////////////////////////////////////
    bool is_complex_vertex(MeshLib::HE_vert<Real> *hv, bool singular_tag);
    ////////////////////////////////////////////////
    bool is_closed_edge_loop(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop);

private:
    MeshLib::Mesh3D<Real> *quad_mesh;
    Real sharp_angle_in_degree;

    std::vector<bool> singular_vertex_tag, boundary_vertex_tag;
    std::vector<bool> complex_edge_tag;

    std::vector<Real> edge_lengths;
    std::vector<ptrdiff_t> face_cluster_id;
    std::vector<bool> corner_tag;
    std::vector<std::vector<MeshLib::HE_edge<Real> *>> complex_edge_loops;
    std::vector<std::vector<ptrdiff_t>> complex_edge_loops_corner_starting_edges;
    std::vector<std::vector<ptrdiff_t>> complex_edge_loops_neighbor_cluster_ids;
    std::vector<ptrdiff_t> complex_edge_loops_cluster_ids;
    std::unordered_map<complex_arc, complex_arc_info> complex_arcs;
    std::vector<Real> group_arc_length;
};