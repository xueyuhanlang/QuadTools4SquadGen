#include "quadimprove.h"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

/////////////////////////////////////////////////////////////////////////////////////////
template <typename Real>
QuadImprove<Real>::QuadImprove(const std::vector<TinyVector<Real, 3>> &input_vertices, const std::vector<std::vector<size_t>> &input_facets)
    : vertices(input_vertices), facets(input_facets)
{
}
template <typename Real>
void QuadImprove<Real>::get_improved_mesh(std::vector<TinyVector<Real, 3>> &output_vertices, std::vector<std::vector<size_t>> &output_facets)
{
    output_vertices = vertices;
    output_facets = facets;
}
/////////////////////////////////////////////////////////////////////////////////////////
template <typename Real>
bool QuadImprove<Real>::improve_mesh()
{
    // create OpenMesh polymesh
    typedef OpenMesh::PolyMesh_ArrayKernelT<> PolyMesh;
    PolyMesh polymesh;
    polymesh.request_vertex_status();
    polymesh.request_edge_status();
    polymesh.request_face_status();

    OpenMesh::VPropHandleT<size_t> v_id_tag;
    OpenMesh::EPropHandleT<bool> edge_feature_tag;
    polymesh.request_face_normals();
    polymesh.add_property(v_id_tag, "v:id_tag");
    polymesh.add_property(edge_feature_tag, "e:feature_tag");

    for (size_t v = 0; v < vertices.size(); v++)
    {
        auto vh = polymesh.add_vertex(OpenMesh::Vec3f((float)vertices[v][0], (float)vertices[v][1], (float)vertices[v][2]));
        polymesh.property(v_id_tag, vh) = v;
    }
    std::vector<PolyMesh::VertexHandle> face_vertices;
    face_vertices.reserve(8);
    for (const auto &f : facets)
    {
        face_vertices.resize(0);
        for (const auto &vid : f)
        {
            face_vertices.push_back(PolyMesh::VertexHandle((int)vid));
        }
        polymesh.add_face(face_vertices);
    }
    polymesh.update_face_normals();
    for (auto e_it = polymesh.edges_begin(); e_it != polymesh.edges_end(); ++e_it)
    {
        auto heh = polymesh.halfedge_handle(*e_it, 0);
        auto fh0 = polymesh.face_handle(heh);
        auto fh1 = polymesh.opposite_face_handle(heh);
        if (polymesh.is_boundary(heh))
        {
            polymesh.property(edge_feature_tag, *e_it) = true;
        }
        else
        {
            auto normal0 = polymesh.normal(fh0);
            auto normal1 = polymesh.normal(fh1);
            auto dot_product = normal0.dot(normal1);
            if (dot_product < 0.866f) // 30 degree
                polymesh.property(edge_feature_tag, *e_it) = true;
            else
                polymesh.property(edge_feature_tag, *e_it) = false;
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////
    // improve quad mesh quality
    bool vertice_moved = false;
    bool faces_changed = false;
    // (1) check interior valence-2 vertices
    for (auto v_it = polymesh.vertices_begin(); v_it != polymesh.vertices_end(); ++v_it)
    {
        if (polymesh.is_boundary(*v_it) || polymesh.valence(*v_it) != 2)
            continue;
        // merge the surrounding two quad faces into one face
        std::vector<PolyMesh::FaceHandle> adjacent_faces;
        for (auto vf_it = polymesh.vf_begin(*v_it); vf_it.is_valid(); ++vf_it)
        {
            adjacent_faces.push_back(*vf_it);
        }
        if (adjacent_faces.size() != 2)
            continue;
        std::vector<PolyMesh::VertexHandle> new_face_vertices;
        new_face_vertices.reserve(4);
        auto heh0 = polymesh.halfedge_handle(*v_it);
        new_face_vertices.push_back(polymesh.to_vertex_handle(heh0));
        auto heh1 = polymesh.next_halfedge_handle(heh0);
        new_face_vertices.push_back(polymesh.to_vertex_handle(heh1));
        auto heh2 = polymesh.next_halfedge_handle(polymesh.opposite_halfedge_handle(heh0));
        new_face_vertices.push_back(polymesh.to_vertex_handle(heh2));
        auto heh3 = polymesh.next_halfedge_handle(heh2);
        new_face_vertices.push_back(polymesh.to_vertex_handle(heh3));

        polymesh.delete_face(adjacent_faces[0], false);
        polymesh.delete_face(adjacent_faces[1], false);
        polymesh.add_face(new_face_vertices);
        polymesh.delete_vertex(*v_it, false);
        vertice_moved = true;
        faces_changed = true;
    }
    ////////////////////////////////////////////////////////////////////////////////////
    // (2) update connectivity
    // for (auto e_it = polymesh.edges_begin(); e_it != polymesh.edges_end(); ++e_it)
    // {
    //     if (!polymesh.is_valid(*e_it) || polymesh.property(edge_feature_tag, *e_it))
    //         continue;
    //     auto heh = polymesh.halfedge_handle(*e_it, 0);
    //     auto fh0 = polymesh.face_handle(heh);
    //     auto fh1 = polymesh.opposite_face_handle(heh);
    //     if (polymesh.is_boundary(heh) || polymesh.is_boundary(polymesh.opposite_halfedge_handle(heh)))
    //         continue;
    //     // check if both faces are quads
    //     if (polymesh.valence(fh0) != 4 || polymesh.valence(fh1) != 4)
    //         continue;
    //     // flip the edge
    //     faces_changed = true;
    // }

    ////////////////////////////////////////////////////////////////////////////////////
    // dump back to vertices and facets
    if (!faces_changed)
        return false;

    polymesh.garbage_collection();

    std::vector<size_t> old_to_new_vertex_map;
    old_to_new_vertex_map.resize(polymesh.n_vertices(), (size_t)(-1));
    size_t new_vid = 0;
    for (auto v_it = polymesh.vertices_begin(); v_it != polymesh.vertices_end(); ++v_it)
    {
        old_to_new_vertex_map[polymesh.property(v_id_tag, *v_it)] = new_vid;
        new_vid++;
    }

    if (vertice_moved)
    {
        vertices.resize(polymesh.n_vertices());
        for (auto v_it = polymesh.vertices_begin(); v_it != polymesh.vertices_end(); ++v_it)
        {
            size_t vid = old_to_new_vertex_map[polymesh.property(v_id_tag, *v_it)];
            auto point = polymesh.point(*v_it);
            vertices[vid][0] = (Real)point[0];
            vertices[vid][1] = (Real)point[1];
            vertices[vid][2] = (Real)point[2];
        }
    }

    facets.resize(0);
    for (auto f_it = polymesh.faces_begin(); f_it != polymesh.faces_end(); ++f_it)
    {
        std::vector<size_t> face;
        for (auto fv_it = polymesh.fv_begin(*f_it); fv_it.is_valid(); ++fv_it)
        {
            size_t vid = polymesh.property(v_id_tag, *fv_it);
            face.push_back(old_to_new_vertex_map[vid]);
        }
        facets.push_back(face);
    }

    return true;
}
/////////////////////////////////////////////////////////////////////////////////////////

// template instantiation
template class QuadImprove<double>;