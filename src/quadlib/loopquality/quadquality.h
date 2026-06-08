#pragma once

#include "Mesh3D.h"
#include <vector>

template <typename Real>
class QuadQuality
{
public:
	QuadQuality(MeshLib::Mesh3D<Real> *mesh, bool _verbose = true);
	void export_base_complex_edges_as_obj(const char filename[]);
	void export_base_complex_faces_as_off(const char filename[]);
	void export_base_complex_edges_as_ply(const char filename[]);
	void export_base_complex_faces_as_ply(const char filename[]);	
	void export_faceloop_quality(const char filename[]);
	void export_edgeloop_quality(const char filename[]);
	void export_quadface_quality(const char filename[]);
	void export_mesh_quality(const char filename[]);
	Real get_simple_faceloop_ratio();
	Real get_simple_edgeloop_ratio();
	Real get_simple_faceloop_ratio_new();
	Real get_simple_edgeloop_ratio_new();
	Real get_faceloop_spriality_ratio();
	Real get_edgeloop_spriality_ratio();
	size_t get_num_of_complex();
	bool is_checkerable();
	size_t get_irregular_vertex_num();
	const std::vector<ptrdiff_t> &get_face_complex_ids() const;
	void get_complex_distribution(Real &max_area_ratio, Real &min_area_ratio, Real &mean_area_ratio, Real &min_edge_length);
	Real get_mean_scaled_jacobian();
protected:
	void compute_quad_quality();
	void compute_mesh_quality();
	void compute_loop_and_layout_quality();
	void edge_loop_quality(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, bool &closed, int &num_self_intersection, Real &rotation_index);
	void compute_rotational_index(const std::vector<TinyVector<Real, 3>> &points, bool closed, Real &rotation_index);
	void export_edge_loop_as_obj(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, const char filename[]);
	void face_loop_quality(const std::vector<MeshLib::HE_face<Real> *> &face_loop, bool &closed, int &num_self_intersection, Real &rotation_index);
	void export_face_loop_as_obj(const std::vector<MeshLib::HE_face<Real> *> &face_loop, const char filename[]);

protected:
	bool verbose;
	MeshLib::Mesh3D<Real> *quad_mesh = nullptr;
	std::vector<bool> boundary_vertex_tag, complex_edge_tag;
	std::vector<ptrdiff_t> face_complex_ids;

	std::vector<Real> planarity, regularity, face_areas;
	ptrdiff_t num_irregular_vertices;

	std::vector<Real> closeness_edge_loops, self_intersection_edge_loops, total_curvature_edge_loops;
	std::vector<Real> closeness_face_loops, self_intersection_face_loops, total_curvature_face_loops;
	std::vector<std::vector<MeshLib::HE_edge<Real> *>> edge_loop_collection;
	std::vector<std::vector<MeshLib::HE_face<Real> *>> face_loop_collection;
};
