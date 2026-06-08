#pragma once

#include "Mesh3D.h"
#include "qcdfcommon.h"
#include "AABB_Tree.h"
#include "AABB_Segment_Tree.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

template <typename Real>
class CDFDCDF2QuadMesh
{
public:
    CDFDCDF2QuadMesh(MeshLib::Mesh3D<Real> *mesh, const std::string &featurefile, bool pattern_subdiv);
    ~CDFDCDF2QuadMesh();
    void set_verbose(bool status);
    void set_debug_mode(int status);
    void set_confusion_band(Real band);
    void set_sharp_angle(const Real angle_in_dgreee);
    void export_mesh(const std::string &filename);
    void set_debug_dir(const std::string &dir);
    void set_ring_size(int size);
    void set_normalize(bool status);
    void set_improve_mode(bool status);
    void set_subdiv_num(unsigned int num);
    void set_smooth_num(unsigned int num);
    void set_edge_collapse_ratio(const Real ratio);
    void set_edge_collapse_normal_threshold(const Real angle_in_degree);

protected:
    bool load_feature(const std::string &featurefile);
    void quad_extraction();

private:
    void init();
    void extract_trimesh_featurelines(std::vector<int> &vertex_feature_tag,
                                      std::vector<std::vector<MeshLib::HE_vert<Real> *>> &feature_edge_loops);

    void extract_trimesh_featurelines_using_cluster(std::vector<int> &vertex_feature_tag,
                                      std::vector<std::vector<MeshLib::HE_vert<Real> *>> &feature_edge_loops);

    void feature_edge_travel(MeshLib::HE_edge<Real> *start_edge,
                             const std::vector<int> &vertex_feature_tag,
                             std::vector<int> &feature_edge_tag,
                             std::vector<MeshLib::HE_vert<Real> *> &loop_vertices);

    void color_pattern_subdivision();

    int cluster_via_polarization();
    void process_uncluster_faces(const int num_clusters);
    bool handle_incorrect_clusters(int &num_clusters);
    void process_unlabeled_faces();
    int reindex_clusters();
    Real distance_to_cluster(const ptrdiff_t fid, const ptrdiff_t cluster_centerface_id);

    void cdf_build_quad_patches();

    bool is_manifold_vertex(const ptrdiff_t vid);
    bool has_nonmanifold_vertex(const ptrdiff_t face_id);

    int identify_seed_faces(std::vector<FaceClusterType> &seed_face_tags);
    void face_clustering_via_seeds(const std::vector<FaceClusterType> &seed_face_tags);

    void merge_clusters();

    int fix_nonquad_faces(const int non_quad_num);

    void update_face_cluster_ids(const std::vector<std::vector<ptrdiff_t>> &face_cluster_ids_store, const int cdf_patch_num);

    void quad_smoothing(
        MeshLib::Mesh3D<Real> *mesh,
        const std::vector<std::set<ptrdiff_t>> &vertex_belonging_clusters,
        const std::unordered_map<ptrdiff_t, std::set<int>> &vertex_to_feature_arc_ids,
        const std::unordered_set<ptrdiff_t> &feature_vertices,
        const std::vector<ig::AABB_Tree<Real> *> &cluster_trees,
        const std::vector<ig::AABB_Segment_Tree<Real> *> &feature_arc_trees);
    void quad_projection(
        MeshLib::Mesh3D<Real> *mesh,
        const std::vector<std::set<ptrdiff_t>> &vertex_belonging_clusters,
        const std::unordered_map<ptrdiff_t, std::set<int>> &vertex_to_feature_arc_ids,
        const std::unordered_set<ptrdiff_t> &feature_vertices,
        const std::vector<ig::AABB_Tree<Real> *> &cluster_trees,
        const std::vector<ig::AABB_Segment_Tree<Real> *> &feature_arc_trees,
        const std::vector<TinyVector<Real, 3>> new_positions,
        const std::vector<int> nonsmooth_tags);

    TinyVector<Real, 3> compute_polyface_normal(const std::vector<TinyVector<Real, 3>> &face_vertices);

protected:
    MeshLib::Mesh3D<Real> *m_pmesh = 0;
    bool m_debug_mode = false, normalize_input = false, improve_mode = true, m_verbose = true;
    int ring_size = 2; // large ring size for better seed merging
    bool pattern_subdivision = false;
    Real global_scale = 1, min_color = 0, max_color = 1, mid_color = (Real)0.5;
    Real confuse_band = (Real)0.1;
    Real sharp_feature_angle = (Real)150;     // in degree
    Real edge_collapse_ratio = (Real)0.1;     // should be between 0 and 1
    Real edge_collapse_normal_threshold = 30; // in degree
    Real feature_color_band = (Real)0.2;
    unsigned int subdiv_num = 0, smooth_num = 0;
    std::string debug_dir = "./";

    std::vector<int> trimesh_feature_edge_tags, color_feature_edge_tags;
    std::vector<Real> face_cdf_colors, face_dcdf_colors, vertex_cdf_colors, vertex_dcdf_colors;
    std::vector<TinyVector<Real, 3>> face_centroids, coffset, doffset, v_coffset, v_doffset;
    std::vector<Real> face_areas;
    TinyVector<Real, 3> inverse_rotation[3], normalization_center;

    std::vector<ptrdiff_t> face_cluster_ids;
    std::vector<ptrdiff_t> cluster_center_face_ids;

    // for feature line extraction
    std::vector<int> vertex_feature_tag;
    std::vector<std::vector<MeshLib::HE_vert<Real> *>> feature_edge_loops;

    // for quad mesh generation
    std::vector<TinyVector<Real, 3>> quad_vertices;
    std::vector<std::vector<size_t>> quad_faces;
};
