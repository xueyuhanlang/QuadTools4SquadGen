#pragma once

#include "Mesh3D.h"
#include "basecomplex.h"

template <typename Real>
class PatchSample
{
public:
    ////////////////////////////////////////////////
    PatchSample(MeshLib::Mesh3D<Real> *input_quad_mesh,
                int num_samples = 50000,
                int num_fps_points_min = 512,
                int num_fps_points_max = 4096,
                int num_fps_copies = 5,
                bool model_normalize = true,
                int random_seed = 0,
                bool use_mesh_as_complex = false,
                Real sharpangle = 120,
                int resolution = -1,
                bool debug = false);
    ////////////////////////////////////////////////
    ~PatchSample();
    ////////////////////////////////////////////////
    void save_samples_to_npz(const std::string &npzfilename, bool quadextractioninfo = false);

protected:
    ////////////////////////////////////////////////
    void compute_patch_distance(bool use_mesh_as_complex, Real sharpangle);
    ////////////////////////////////////////////////
    // void save_complex_edges(const std::string &obj_filename, MeshLib::Mesh3D<Real> *mesh, const std::vector<bool> &complex_edge_tag);
    ////////////////////////////////////////////////
    void decompose_subdiv_mesh();
    ////////////////////////////////////////////////
    void
    get_grading_color_all(const int findex, const TinyVector<Real, 3> &sample_point,
                          Real &color0, Real &color1, TinyVector<Real, 3> *color_gradient0 = 0, TinyVector<Real, 3> *color_gradient1 = 0);
    ////////////////////////////////////////////////
    Real get_grading_color(const int findex, const TinyVector<Real, 3> &sample_point, TinyVector<Real, 3> *color_gradient = 0);
    ////////////////////////////////////////////////
    void set_mesh_as_complex(MeshLib::Mesh3D<Real> *mesh,
                             std::vector<bool> &complex_edge_tag,
                             std::vector<bool> &corner_tag,
                             std::vector<std::vector<MeshLib::HE_edge<Real> *>> &complex_edge_loops,
                             std::vector<std::vector<ptrdiff_t>> &complex_edge_loops_corner_starting_edges,
                             std::vector<std::vector<ptrdiff_t>> &complex_edge_loops_neighbor_cluster_ids,
                             std::vector<ptrdiff_t> &complex_edge_loops_cluster_ids,
                             std::unordered_map<complex_arc, complex_arc_info> &complex_arcs,
                             size_t &arc_group_num);
    ////////////////////////////////////////////////
protected:
    bool valid = false;
    MeshLib::Mesh3D<Real> *quad_mesh = 0, *subdiv_mesh = 0, *dualquad_mesh = 0;
    bool debug_mode = false;
    int div = 1;
    std::vector<Real> subdiv_mesh_vertex_color;
    std::vector<bool> tag_wall_edges, split_flip;
    std::vector<ptrdiff_t> edge_vertex_color, dualquad_vertex_id_map;
    std::vector<std::pair<Real, Real>> edge_color_store;

    int num_dual_patches = 0, num_quads = 0;
    std::vector<ptrdiff_t> quad_id_map;
    std::vector<ptrdiff_t> quad2patches;

    std::vector<TinyVector<Real, 3>> sample_points;
    std::vector<ptrdiff_t> sample_point_face_ids;
    std::vector<TinyVector<Real, 3>> sample_point_bary_coords;

    std::vector<int> num_fps_points_list;
    std::unordered_map<int, std::vector<std::vector<TinyVector<Real, 3>>>> fps_points_mp;
    std::unordered_map<int, std::vector<std::vector<TinyVector<Real, 3>>>> fps_bary_coords_mp;
    std::unordered_map<int, std::vector<std::vector<ptrdiff_t>>> fps_point_face_ids_mp;

    // scale and PCA parameters
    TinyVector<Real, 3> normalization_center, inverse_rotation[3];
    Real normalization_scale = (Real)1.0;

    // mesh info
    int num_complex = 0, num_boundary = 0, num_singularity = 0;
    // Real loop_score = 0;
};