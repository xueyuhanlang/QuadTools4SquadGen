#include "basecomplex.h"
#include <omp.h>
#include <algorithm>
#include "looptravel.h"
#include "looputil.h"
#include "happly.h"
#include <stdexcept>
#include <queue>
#include "MeshWriter.h"

complex_arc::complex_arc(const ptrdiff_t vstart_id, const ptrdiff_t vend_id, const ptrdiff_t cluster_id0, const ptrdiff_t cluster_id1)
{
    end_vertices[0] = vstart_id, end_vertices[1] = vend_id;
    cluster_id[0] = cluster_id0, cluster_id[1] = cluster_id1;
    if (vstart_id > vend_id)
    {
        std::swap(end_vertices[0], end_vertices[1]);
        std::swap(cluster_id[0], cluster_id[1]);
    }
}

bool complex_arc::operator==(const complex_arc &c) const
{
    for (int i = 0; i < 2; i++)
    {
        if (end_vertices[i] != c.end_vertices[i] || cluster_id[i] != c.cluster_id[i])
        {
            return false;
        }
    }

    return true;
}
bool complex_arc::operator!=(const complex_arc &c) const
{
    for (int i = 0; i < 2; i++)
    {
        if (end_vertices[i] != c.end_vertices[i] || cluster_id[i] != c.cluster_id[i])
        {
            return true;
        }
    }
    return false;
}
bool complex_arc::operator<(const complex_arc &c) const
{
    for (int i = 0; i < 2; i++)
    {
        if (end_vertices[i] < c.end_vertices[i])
        {
            return true;
        }
        else if (end_vertices[i] > c.end_vertices[i])
        {
            return false;
        }
    }
    for (int i = 0; i < 2; i++)
    {
        if (cluster_id[i] < c.cluster_id[i])
        {
            return true;
        }
        else if (cluster_id[i] > c.cluster_id[i])
        {
            return false;
        }
    }
    return false;
}

complex_arc_info::complex_arc_info()
{
    ring_neighbor_arc_1[0] = ring_neighbor_arc_1[1] = ring_neighbor_arc_2[0] = ring_neighbor_arc_2[1] = 0;
    cluster_id_1[0] = cluster_id_1[1] = cluster_id_2[0] = cluster_id_2[1] = -1;
    arclength = 0;
    visited = false;
    group_id = -1;
    orientation = true;
}
bool complex_arc_info::add_ring_neighbor(const ptrdiff_t v1, const ptrdiff_t v2, ptrdiff_t cluster_id0, ptrdiff_t cluster_id1)
{
    auto tmp_v1 = v1, tmp_v2 = v2;
    auto tmp_cid0 = cluster_id0, tmp_cid1 = cluster_id1;
    if (v1 > v2)
    {
        std::swap(tmp_v1, tmp_v2);
        std::swap(tmp_cid0, tmp_cid1);
    }

    if (ring_neighbor_arc_1[0] == 0 && ring_neighbor_arc_1[1] == 0 && cluster_id_1[0] == -1 && cluster_id_1[1] == -1)
    {
        ring_neighbor_arc_1[0] = tmp_v1, ring_neighbor_arc_1[1] = tmp_v2;
        cluster_id_1[0] = tmp_cid0, cluster_id_1[1] = tmp_cid1;
        return true;
    }
    else if (ring_neighbor_arc_1[0] == tmp_v1 && ring_neighbor_arc_1[1] == tmp_v2 && cluster_id_1[0] == tmp_cid0 && cluster_id_1[1] == tmp_cid1)
    {
        return false;
    }

    if (ring_neighbor_arc_2[0] == 0 && ring_neighbor_arc_2[1] == 0 && cluster_id_2[0] == -1 && cluster_id_2[1] == -1)
    {
        ring_neighbor_arc_2[0] = tmp_v1, ring_neighbor_arc_2[1] = tmp_v2;
        cluster_id_2[0] = tmp_cid0, cluster_id_2[1] = tmp_cid1;
        return true;
    }
    else if (ring_neighbor_arc_2[0] == tmp_v1 && ring_neighbor_arc_2[1] == tmp_v2 && cluster_id_2[0] == tmp_cid0 && cluster_id_2[1] == tmp_cid1)
    {
        return false;
    }
    std::cout << "more than two ring neighbors for one arc!" << std::endl;
    throw std::runtime_error("Error: more than two ring neighbors for one arc! <basecomplex.cpp:add_ring_neighbor>");
    return false;
}
///////////////////////////////////////////////////
template <typename Real>
BaseComplex<Real>::BaseComplex(MeshLib::Mesh3D<Real> *input_mesh, Real sharp_angle)
    : quad_mesh(input_mesh), sharp_angle_in_degree(sharp_angle)
{
    if (!quad_mesh)
    {
        std::cout << "Error: Empty input!" << std::endl;
        return;
    }
    else if (!quad_mesh->is_quad())
    {
        std::cout << "Error: The input mesh is not a quad mesh!" << std::endl;
        return;
    }

    pre_computation();
    extract_base_complex();
}
///////////////////////////////////////////////
template <typename Real>
const std::vector<bool> &BaseComplex<Real>::get_boundary_vertex_tag() const
{
    return boundary_vertex_tag;
}
///////////////////////////////////////////////
template <typename Real>
const std::vector<bool> &BaseComplex<Real>::get_singular_vertex_tag() const
{
    return singular_vertex_tag;
}
///////////////////////////////////////////////
template <typename Real>
const std::vector<bool> &BaseComplex<Real>::get_complex_edge_tag() const
{
    return complex_edge_tag;
}
/////////////////////////////////////////////////
template <typename Real>
const std::vector<bool> &BaseComplex<Real>::get_corner_tag() const
{
    return corner_tag;
}
/////////////////////////////////////////////////
template <typename Real>
const std::vector<ptrdiff_t> &BaseComplex<Real>::get_face_patch_ids() const
{
    return face_cluster_id;
}
/////////////////////////////////////////////////
template <typename Real>
const std::vector<std::vector<MeshLib::HE_edge<Real> *>> &BaseComplex<Real>::get_complex_edge_loops() const
{
    return complex_edge_loops;
}
/////////////////////////////////////////////////
template <typename Real>
const std::vector<std::vector<ptrdiff_t>> &BaseComplex<Real>::get_complex_edge_loops_corner_starting_edges() const
{
    return complex_edge_loops_corner_starting_edges;
}
/////////////////////////////////////////////////
template <typename Real>
const std::vector<std::vector<ptrdiff_t>> &BaseComplex<Real>::get_complex_edge_loops_neighbor_cluster_ids() const
{
    return complex_edge_loops_neighbor_cluster_ids;
}
/////////////////////////////////////////////////
template <typename Real>
const std::vector<ptrdiff_t> &BaseComplex<Real>::get_complex_edge_loops_cluster_ids() const
{
    return complex_edge_loops_cluster_ids;
}
template <typename Real>
const std::unordered_map<complex_arc, complex_arc_info> &BaseComplex<Real>::get_complex_arcs() const
{
    return complex_arcs;
}
template <typename Real>
const size_t BaseComplex<Real>::get_arc_group_num() const
{
    return group_arc_length.size();
}
/////////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::export_complex_as_ply(const char filename[], bool save_curved_edges)
{
    std::vector<float> vertexX, vertexY, vertexZ;
    std::vector<std::vector<size_t>> faceIndices;
    faceIndices.reserve(complex_edge_loops.size());
    std::unordered_map<ptrdiff_t, size_t> vertex_id_map;
    size_t vertex_count = 0;

    if (!save_curved_edges)
    {
        for (size_t i = 0; i < corner_tag.size(); i++)
        {
            if (corner_tag[i])
            {
                auto vert = quad_mesh->get_vertex(i);
                vertexX.emplace_back((float)vert->pos[0]);
                vertexY.emplace_back((float)vert->pos[1]);
                vertexZ.emplace_back((float)vert->pos[2]);
                vertex_id_map[i] = vertex_count++;
            }
        }

        std::vector<size_t> face(4);
        for (size_t i = 0; i < complex_edge_loops.size(); i++)
        {
            const auto &boundary_edge_loop = complex_edge_loops[i];
            const auto &corner_indices_on_loop = complex_edge_loops_corner_starting_edges[i];

            for (int j = 0; j < 4; j++)
                face[j] = vertex_id_map[boundary_edge_loop[corner_indices_on_loop[j]]->pair->vert->id];
            faceIndices.emplace_back(face);
        }
    }
    else
    {
        for (size_t i = 0; i < complex_edge_loops.size(); i++)
        {
            const auto &boundary_edge_loop = complex_edge_loops[i];

            std::vector<size_t> face;
            for (size_t j = 0; j < boundary_edge_loop.size(); j++)
            {
                auto vert = boundary_edge_loop[j]->pair->vert;
                auto iter = vertex_id_map.find(vert->id);
                if (iter == vertex_id_map.end())
                {
                    vertexX.emplace_back((float)vert->pos[0]);
                    vertexY.emplace_back((float)vert->pos[1]);
                    vertexZ.emplace_back((float)vert->pos[2]);
                    vertex_id_map[vert->id] = vertex_count++;
                    face.emplace_back(vertex_count - 1);
                }
                else
                {
                    face.emplace_back(iter->second);
                }
            }
            faceIndices.emplace_back(face);
        }
    }

    happly::PLYData plyOut;
    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addFaceIndices(faceIndices);

    plyOut.write(filename, happly::DataFormat::Binary);
}
/////////////////////////////////////////////////
template <typename Real>
const int BaseComplex<Real>::get_num_singularity() const
{
    int num_singularities = 0;
#pragma omp parallel for reduction(+ : num_singularities)
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        if (singular_vertex_tag[i])
            num_singularities++;
    }
    return num_singularities;
}
template <typename Real>
const int BaseComplex<Real>::get_num_complex() const
{
    // note: the complex is the splitted version
    return (int)complex_edge_loops.size();
}
/////////////////////////////////////////////////
template <typename Real>
bool BaseComplex<Real>::is_closed_edge_loop(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop)
{
    return edge_loop.size() > 2 && edge_loop.front()->pair->vert == edge_loop.back()->vert;
}
/////////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::extract_base_complex()
{

    quad_mesh->reset_edges_tag(false);
    complex_edge_tag.assign(quad_mesh->get_num_of_edges(), false);

    ///////////////////////////////////////////////////
    std::vector<std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real>> all_close_loops_with_length;

    // travel edge loops starting from irregular vertices
    int counter = 0;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = quad_mesh->get_vertex(i);
        if (!singular_vertex_tag[i])
            continue;
        auto he = hv->edge;
        do
        {
            if (he->tag || he->pair->tag)
                ;
            else
            {
                std::vector<MeshLib::HE_edge<Real> *> edge_loop;
                edge_loop_travel(he, edge_loop, singular_vertex_tag, complex_edge_tag, true);
                if (is_closed_edge_loop(edge_loop))
                {
                    // SaveEdge_as_ply<double>("loop_" + std::to_string(counter++) + ".ply", edge_loop);
                    all_close_loops_with_length.emplace_back(
                        std::make_pair(edge_loop, get_edgeloop_length(edge_loop)));
                }
            }
            he = he->pair->next;
        } while (he != hv->edge);
    }

    // travel boundary edge loops
    std::vector<std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real>> boundary_edge_loops_with_length;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge->tag || edge->pair->tag || (edge->face && edge->pair->face))
            continue;
        std::vector<MeshLib::HE_edge<Real> *> edge_loop;
        edge_loop_travel(edge, edge_loop, singular_vertex_tag, complex_edge_tag, true);
        if (is_closed_edge_loop(edge_loop))
        {
            // SaveEdge_as_ply<double>("./bloop" + std::to_string(counter++) + ".ply", edge_loop);

            boundary_edge_loops_with_length.emplace_back(
                std::make_pair(edge_loop, get_edgeloop_length(edge_loop)));
        }
        else
        {
            std::cout << "should not happen: boundary edge loop is not closed!" << std::endl;
            throw std::runtime_error("Error: boundary edge loop is not closed! <basecomplex.cpp:extract_base_complex>");
        }
    }

    // process sharp edge loops
    Real edge_length_ratio_threshold = (Real)0.5;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (complex_edge_tag[i] || edge->tag || edge->pair->tag || quad_mesh->is_on_boundary(edge))
            continue;
        if (compute_dihedral_angle(edge) < sharp_angle_in_degree)
        {
            std::vector<MeshLib::HE_edge<Real> *> interior_edge_loop;
            edge_loop_travel(edge, interior_edge_loop, singular_vertex_tag, complex_edge_tag, true);
            Real looplength = get_edgeloop_length(interior_edge_loop);
            Real sharpedgelength = 0;
            for (size_t j = 0; j < interior_edge_loop.size(); j++)
            {
                if (compute_dihedral_angle(interior_edge_loop[j]) < sharp_angle_in_degree)
                    sharpedgelength += edge_lengths[interior_edge_loop[j]->id];
            }
            if (sharpedgelength >= edge_length_ratio_threshold * looplength)
            {
                if (is_closed_edge_loop(interior_edge_loop))
                {
                    all_close_loops_with_length.emplace_back(std::make_pair(interior_edge_loop, looplength));
                }
            }
            else
            {
                tag_edge_loop_for_split(interior_edge_loop, false);
            }
        }
    }

    // check whether there is no singularity
    bool has_singularity = false;
    for (auto tag : singular_vertex_tag)
    {
        if (tag)
        {
            has_singularity = true;
            break;
        }
    }

    // split boundary edge loops to ensure that they are dvided into more than 1 segment
    split_edge_loop(boundary_edge_loops_with_length, has_singularity);

    // split closed sharp edge loops to ensure that they are dvided into more than 1 segment
    split_edge_loop(all_close_loops_with_length, has_singularity);

    // check validity of each noncomplex closed edge loop, whether it is divided into more than 1 segment
    all_close_loops_with_length.resize(0);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge->tag || edge->pair->tag || complex_edge_tag[i])
            continue;
        std::vector<MeshLib::HE_edge<Real> *> interior_edge_loop;
        edge_loop_travel(edge, interior_edge_loop, singular_vertex_tag, complex_edge_tag, false);
        if (is_closed_edge_loop(interior_edge_loop))
        {
            Real looplength = get_edgeloop_length(interior_edge_loop);
            all_close_loops_with_length.emplace_back(
                std::make_pair(interior_edge_loop, looplength));
        }
    }
    for (auto &edge_loop : all_close_loops_with_length)
    {
        for (auto &e : edge_loop.first)
        {
            e->tag = e->pair->tag = false;
        }
    }
    // ensure that non-complex closed edge loops are dvided into more than 1 segment
    split_edge_loop(all_close_loops_with_length, has_singularity);

    // prepare information for patchsample
    // extract base complex (four corners, four start edges)
    corner_tag.assign(quad_mesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        corner_tag[i] = is_complex_corner(quad_mesh->get_vertex(i), singular_vertex_tag[i]);
    }

    compute_base_patch();
}
////////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::pre_computation()
{
    singular_vertex_tag.assign(quad_mesh->get_num_of_vertices(), false);
    boundary_vertex_tag.assign(quad_mesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = quad_mesh->get_vertex(i);
        boundary_vertex_tag[i] = quad_mesh->is_on_boundary(hv);
        singular_vertex_tag[i] = boundary_vertex_tag[i] ? hv->degree != 3 : hv->degree != 4;
    }

    edge_lengths.resize(quad_mesh->get_num_of_edges());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge > edge->pair)
            continue;
        edge_lengths[i] = edge_lengths[edge->pair->id] = edge->GetLength();
    }
}
////////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::compute_base_patch()
{
    face_cluster_id.assign(quad_mesh->get_num_of_faces(), -1);
    std::queue<MeshLib::HE_face<Real> *> face_queue;
    int cluster_counter = 0;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        if (face_cluster_id[i] == -1)
        {
            face_cluster_id[i] = cluster_counter;
            face_queue.push(quad_mesh->get_face(i));
            while (!face_queue.empty())
            {
                auto f = face_queue.front();
                face_queue.pop();
                auto he = f->edge;
                do
                {
                    if (he->pair->face && face_cluster_id[he->pair->face->id] == -1 && !complex_edge_tag[he->id])
                    {
                        face_cluster_id[he->pair->face->id] = cluster_counter;
                        face_queue.push(he->pair->face);
                    }
                    he = he->next;
                } while (he != f->edge);
            }
            cluster_counter++;
        }
    }
    /// extract complex edge loops
    quad_mesh->reset_edges_tag(false);

    complex_edge_loops.resize(0);
    complex_edge_loops_corner_starting_edges.resize(0);
    complex_edge_loops_neighbor_cluster_ids.resize(0);
    complex_edge_loops_cluster_ids.resize(0);

    for (size_t i = 0; i < complex_edge_tag.size(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (complex_edge_tag[i] && edge->tag == false)
        {
            if (!edge->face)
                continue;

            complex_edge_loops_cluster_ids.emplace_back(face_cluster_id[edge->face->id]);
            std::vector<MeshLib::HE_edge<Real> *> boundary_edge_loop;
            MeshLib::HE_vert<Real> *vstart = edge->pair->vert;
            ptrdiff_t corner_indices_on_loop[4] = {-1, -1, -1, -1};
            int corner_count = 0;
            std::vector<ptrdiff_t> neighbor_cluster_ids;
            while (1)
            {
                boundary_edge_loop.emplace_back(edge);
                if (corner_tag[edge->pair->vert->id])
                {
                    if (corner_count >= 4)
                    {
                        std::cout << "error (corner_count " << corner_count << " >= 4)" << std::endl;
                        throw std::runtime_error("Error: more than four corners on one complex edge loop! <basecomplex.cpp:extract_base_complex>");
                    }
                    else
                    {
                        neighbor_cluster_ids.emplace_back(edge->pair->face ? face_cluster_id[edge->pair->face->id] : -1);
                        corner_indices_on_loop[corner_count] = boundary_edge_loop.size() - 1;
                        corner_count++;
                    }
                }
                if (corner_tag[edge->vert->id])
                    edge = edge->next;
                else
                    edge = edge->next->pair->next;
                if (edge == boundary_edge_loop.front())
                    break;
            }

            complex_edge_loops.emplace_back(boundary_edge_loop);
            complex_edge_loops_corner_starting_edges.push_back({corner_indices_on_loop[0], corner_indices_on_loop[1], corner_indices_on_loop[2], corner_indices_on_loop[3]});
            complex_edge_loops_neighbor_cluster_ids.emplace_back(neighbor_cluster_ids);
            for (const auto &e : boundary_edge_loop)
            {
                e->tag = true;
            }
        }
    }

    complex_arcs.clear();
    for (size_t i = 0; i < complex_edge_loops.size(); i++)
    {
        const auto &boundary_edge_loop = complex_edge_loops[i];
        const auto &corner_indices_on_loop = complex_edge_loops_corner_starting_edges[i];

        MeshLib::HE_vert<Real> *corner_verts[4] = {boundary_edge_loop[corner_indices_on_loop[0]]->pair->vert, boundary_edge_loop[corner_indices_on_loop[1]]->pair->vert, boundary_edge_loop[corner_indices_on_loop[2]]->pair->vert, boundary_edge_loop[corner_indices_on_loop[3]]->pair->vert};

        for (int j = 0; j < 4; j++)
        {
            auto start_edge = boundary_edge_loop[corner_indices_on_loop[j]];
            auto end_edge = boundary_edge_loop[corner_indices_on_loop[(j + 1) % 4]];
            auto start_vert = start_edge->pair->vert;
            auto end_vert = end_edge->pair->vert;

            complex_arc arc(start_vert->id, end_vert->id, complex_edge_loops_cluster_ids[i], complex_edge_loops_neighbor_cluster_ids[i][j]);
            auto iter = complex_arcs.find(arc);
            if (iter != complex_arcs.end())
            {
                iter->second.add_ring_neighbor(corner_verts[(j + 2) % 4]->id, corner_verts[(j + 3) % 4]->id, complex_edge_loops_cluster_ids[i], complex_edge_loops_neighbor_cluster_ids[i][(j + 2) % 4]);
            }
            else
            {
                complex_arc_info arcinfo;
                arcinfo.add_ring_neighbor(corner_verts[(j + 2) % 4]->id, corner_verts[(j + 3) % 4]->id, complex_edge_loops_cluster_ids[i], complex_edge_loops_neighbor_cluster_ids[i][(j + 2) % 4]);
                arcinfo.arclength = 0;
                for (size_t k = 0; k < boundary_edge_loop.size(); k++)
                {
                    auto edge = boundary_edge_loop[(k + corner_indices_on_loop[j]) % boundary_edge_loop.size()];
                    if (edge == end_edge)
                        break;
                    arcinfo.arclength += (double)edge_lengths[edge->id];
                }
                complex_arcs[arc] = arcinfo;
            }
        }
    }

    group_arc_length.resize(0);

    int group_count = 0;
    for (auto &c_arc : complex_arcs)
    {
        if (c_arc.second.visited)
            continue;
        std::queue<complex_arc> arc_queue;
        arc_queue.push(c_arc.first);

        c_arc.second.orientation = true;

        Real avg_arc_length = 0;

        int edge_count = 0;
        while (!arc_queue.empty())
        {
            auto arc = arc_queue.front();
            arc_queue.pop();
            auto iter = complex_arcs.find(arc);
            if (iter->second.visited)
                continue;

            bool orientation = iter->second.orientation;
            iter->second.visited = true;
            iter->second.group_id = group_count;

            avg_arc_length += iter->second.arclength;
            edge_count++;

            if (iter->second.ring_neighbor_arc_1[0] != -1 && iter->second.ring_neighbor_arc_1[1] != -1)
            {
                complex_arc arc1(iter->second.ring_neighbor_arc_1[0], iter->second.ring_neighbor_arc_1[1], iter->second.cluster_id_1[0], iter->second.cluster_id_1[1]);
                auto iter1 = complex_arcs.find(arc1);
                if (iter1 != complex_arcs.end() && !iter1->second.visited)
                {
                    arc_queue.push(arc1);

                    if (iter->first.cluster_id[0] != -1 && iter->first.cluster_id[0] == iter1->first.cluster_id[0])
                        iter1->second.orientation = !orientation;
                    else if (iter->first.cluster_id[0] != -1 && iter->first.cluster_id[0] == iter1->first.cluster_id[1])
                        iter1->second.orientation = orientation;
                    else if (iter->first.cluster_id[1] != -1 && iter->first.cluster_id[1] == iter1->first.cluster_id[0])
                        iter1->second.orientation = orientation;
                    else if (iter->first.cluster_id[1] != -1 && iter->first.cluster_id[1] == iter1->first.cluster_id[1])
                        iter1->second.orientation = !orientation;
                    else
                    {
                        throw std::runtime_error("Error: cannot determine the orientation of arcs! <basecomplex.cpp:compute_base_patch>");
                    }
                }
            }

            if (iter->second.ring_neighbor_arc_2[0] != -1 && iter->second.ring_neighbor_arc_2[1] != -1)
            {
                complex_arc arc2(iter->second.ring_neighbor_arc_2[0], iter->second.ring_neighbor_arc_2[1], iter->second.cluster_id_2[0], iter->second.cluster_id_2[1]);
                auto iter2 = complex_arcs.find(arc2);
                if (iter2 != complex_arcs.end() && !iter2->second.visited)
                {
                    arc_queue.push(arc2);
                    if (iter->first.cluster_id[0] != -1 && iter->first.cluster_id[0] == iter2->first.cluster_id[0])
                        iter2->second.orientation = !orientation;
                    else if (iter->first.cluster_id[0] != -1 && iter->first.cluster_id[0] == iter2->first.cluster_id[1])
                        iter2->second.orientation = orientation;
                    else if (iter->first.cluster_id[1] != -1 && iter->first.cluster_id[1] == iter2->first.cluster_id[0])
                        iter2->second.orientation = orientation;
                    else if (iter->first.cluster_id[1] != -1 && iter->first.cluster_id[1] == iter2->first.cluster_id[1])
                        iter2->second.orientation = !orientation;
                    else
                    {
                        throw std::runtime_error("Error: cannot determine the orientation of arcs! <basecomplex.cpp:compute_base_patch>");
                    }
                }
            }
        }
        group_arc_length.emplace_back(avg_arc_length / edge_count);
        group_count++;
    }
}
////////////////////////////////////////////////
template <typename Real>
Real BaseComplex<Real>::get_edgeloop_length(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop)
{
    Real length = 0;
#pragma omp parallel for reduction(+ : length)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)edge_loop.size(); i++)
    {
        length += edge_lengths[edge_loop[i]->id];
    }
    return length;
}
////////////////////////////////////////////////
template <typename Real>
int BaseComplex<Real>::count_complex_vertices(const std::vector<MeshLib::HE_edge<Real> *> &close_edge_loop, std::vector<int> &count_corners)
{
    count_corners.resize(0);
    int pos = 0;
    for (auto e : close_edge_loop)
    {
        auto v = e->pair->vert;
        auto edge = v->edge;
        int count = 0;
        do
        {
            if (complex_edge_tag[edge->id])
            {
                count++;
            }
            edge = edge->pair->next;
        } while (edge != v->edge);

        if (count >= 3)
            count_corners.emplace_back(pos);
        pos++;
    }
    return (int)count_corners.size();
}
//////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::loop_travel_for_split(MeshLib::HE_vert<Real> *vert, std::vector<MeshLib::HE_edge<Real> *> &edge_loop)
{
    auto edge = vert->edge;
    do
    {
        if (!complex_edge_tag[edge->id])
        {
            edge_loop_travel(edge, edge_loop, singular_vertex_tag, complex_edge_tag, true);
            break;
        }
        edge = edge->pair->next;
    } while (edge != vert->edge);
}
//////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::tag_edge_loop_for_split(std::vector<MeshLib::HE_edge<Real> *> &edge_loop, bool status)
{
    for (auto &e : edge_loop)
    {
        e->tag = e->pair->tag = complex_edge_tag[e->id] = complex_edge_tag[e->pair->id] = status;
    }
}
//////////////////////////////////////////////
template <typename Real>
int BaseComplex<Real>::find_mid_point(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, Real arclength, int start_pos, int end_pos)
{
    Real len = 0;
    auto pos = end_pos;
    for (int i = 0; i < (int)edge_loop.size(); i++)
    {
        auto p = (i + start_pos) % (int)edge_loop.size();
        len += edge_lengths[edge_loop[p]->id];
        if (len >= arclength / 2)
        {
            pos = p;
            break;
        }
        if (p == end_pos)
            break;
    }
    if (pos == end_pos)
        pos = (int)((end_pos - 1 + edge_loop.size()) % edge_loop.size());
    return pos;
}
//////////////////////////////////////////////
template <typename Real>
int BaseComplex<Real>::find_farest_point(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, Real looplength, int start_pos)
{
    Real len = 0;
    int pos = start_pos;
    for (ptrdiff_t i = 0; i < (ptrdiff_t)edge_loop.size(); i++)
    {
        pos = (i + start_pos) % (int)edge_loop.size();
        len += edge_lengths[edge_loop[pos]->id];
        if (len >= looplength / 2)
            break;
    }
    if (pos == start_pos)
        pos = (int)((pos - 1 + edge_loop.size()) % edge_loop.size());
    return pos;
}
//////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::find_two_farest_points(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, Real looplength, int start_pos, int &pos0, int &pos1)
{
    Real len = 0;
    pos0 = start_pos + 1, pos1 = (int)((start_pos - 1 + edge_loop.size()) % edge_loop.size());
    bool found_pos0 = false;
    for (ptrdiff_t i = 0; i < (ptrdiff_t)edge_loop.size(); i++)
    {
        auto pos = (i + start_pos) % (int)edge_loop.size();
        len += edge_lengths[edge_loop[pos]->id];
        if (len >= looplength / 3 && !found_pos0)
        {
            pos0 = (int)pos;
            found_pos0 = true;
        }
        if (len >= 2 * looplength / 3)
        {
            pos1 = (int)pos;
            break;
        }
    }
    if (pos0 == start_pos)
        pos1 = (int)((start_pos - 2 + edge_loop.size()) % edge_loop.size());
    if (pos1 == start_pos)
        pos1 = (int)((start_pos - 1 + edge_loop.size()) % edge_loop.size());
    if (pos0 == pos1)
    {
        auto new_pos0 = (int)((pos0 - 1 + edge_loop.size()) % edge_loop.size());
        if (new_pos0 != start_pos)
            pos0 = (int)new_pos0;
        else
            pos1 = (int)((pos1 + 1) % edge_loop.size());
    }
}
//////////////////////////////////////////////
template <typename Real>
void BaseComplex<Real>::split_edge_loop(std::vector<std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real>> &edge_loops_with_length, bool has_singularity)
{
    if (edge_loops_with_length.empty())
        return;
    std::vector<size_t> original_edge_loop_sizes(edge_loops_with_length.size());
    for (size_t i = 0; i < edge_loops_with_length.size(); i++)
    {
        original_edge_loop_sizes[i] = i;
    }
    // sort the edge loops according to their lengths, from long to short, store the index in original_edge_loop_sizes
    std::sort(original_edge_loop_sizes.begin(), original_edge_loop_sizes.end(),
              [&edge_loops_with_length](size_t a, size_t b)
              {
                  return edge_loops_with_length[a].second > edge_loops_with_length[b].second;
              });
    std::queue<std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real>> edge_loop_queue;
    std::vector<MeshLib::HE_edge<Real> *> new_edge_loop;
    size_t num_process_loops = 0;

    // int loop_split_num = 1;
    // if (has_singularity == false)
    //     loop_split_num = 2; // quad_mesh->is_closed() ? 3 : 2; // 2 is possible; 1 is not possible for twisted rings
    while (true)
    {
        std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real> edge_loop_pair;
        if (!edge_loop_queue.empty())
        {
            edge_loop_pair = edge_loop_queue.front();
            edge_loop_queue.pop();
        }
        else if (num_process_loops < original_edge_loop_sizes.size())
        {
            edge_loop_pair = edge_loops_with_length[original_edge_loop_sizes[num_process_loops]];
            num_process_loops++;
        }
        else
            break;

        auto edge_loop = edge_loop_pair.first;
        auto looplength = edge_loop_pair.second;

        int count_singularities = 0;
        bool boundary_loop = false;
        for (auto e : edge_loop)
        {
            if (singular_vertex_tag[e->pair->vert->id])
                count_singularities++;
        }
        for (auto e : edge_loop)
        {
            if (boundary_vertex_tag[e->pair->vert->id])
            {
                boundary_loop = true;
                break;
            }
        }

        int loop_split_num = 1;
        if (boundary_loop || count_singularities > 0 || !has_singularity)
        {
            loop_split_num = 2;
        }

        bool stop = false;
        while (!stop)
        {
            std::vector<int> count_corners;
            if (count_complex_vertices(edge_loop, count_corners) >= loop_split_num)
                break;

            if (!complex_edge_tag[edge_loop.front()->id])
            {
                // fake the edge loop as complex edge loop, to count the complex vertices
                tag_edge_loop_for_split(edge_loop, true);
                if (count_complex_vertices(edge_loop, count_corners) >= loop_split_num)
                {
                    // reset the tags, no need to split
                    tag_edge_loop_for_split(edge_loop, false);
                    break;
                }
            }
            if (count_corners.size() == 2)
            {
                if (loop_split_num == 2)
                    break;
                // insert a new complex vertex at the middle of the longer arc between the two complex vertices
                Real arclength01 = 0, arclength10 = 0;
                int edgenum01 = 0, edgenum10 = 0;
                for (ptrdiff_t i = 0; i < (ptrdiff_t)edge_loop.size(); i++)
                {
                    auto p = (i + count_corners[0]) % (int)edge_loop.size();
                    if (p == count_corners[1])
                        break;
                    arclength01 += edge_lengths[edge_loop[p]->id];
                    edgenum01++;
                }
                arclength10 = looplength - arclength01;
                edgenum10 = (int)edge_loop.size() - edgenum01;
                int pos = 0;
                if (arclength01 >= arclength10 && edgenum01 > 1)
                {
                    pos = find_mid_point(edge_loop, arclength01, count_corners[0], count_corners[1]);
                }
                else if (arclength10 >= arclength01 && edgenum10 > 1)
                {
                    pos = find_mid_point(edge_loop, arclength10, count_corners[1], count_corners[0]);
                }
                else if (edgenum01 >= edgenum10)
                {
                    pos = find_mid_point(edge_loop, arclength01, count_corners[0], count_corners[1]);
                }
                else
                {
                    pos = find_mid_point(edge_loop, arclength10, count_corners[1], count_corners[0]);
                }
                loop_travel_for_split(edge_loop[pos]->vert, new_edge_loop);
                if (is_closed_edge_loop(new_edge_loop))
                    edge_loop_queue.push(std::make_pair(new_edge_loop, get_edgeloop_length(new_edge_loop)));
                break;
            }
            else if (count_corners.size() == 1) // find the farthest vertex to split
            {
                // first trial: split at the farest vertex from the only complex vertex, and check whether the introducted loops would split the edge_loop into more than 2 segments;
                int pos = find_farest_point(edge_loop, looplength, count_corners[0]);
                loop_travel_for_split(edge_loop[pos]->vert, new_edge_loop);
                std::vector<int> new_count_corners;
                if (loop_split_num == 2 || count_complex_vertices(edge_loop, new_count_corners) >= loop_split_num)
                {
                    if (is_closed_edge_loop(new_edge_loop))
                        edge_loop_queue.push(std::make_pair(new_edge_loop, get_edgeloop_length(new_edge_loop)));
                    break;
                }
                else
                {
                    tag_edge_loop_for_split(new_edge_loop, false);
                }
                // second trial: split the edge_loop into two arcs
                int pos0 = 0, pos1 = 0;
                find_two_farest_points(edge_loop, looplength, count_corners[0], pos0, pos1);
                loop_travel_for_split(edge_loop[pos0]->vert, new_edge_loop);
                if (is_closed_edge_loop(new_edge_loop))
                    edge_loop_queue.push(std::make_pair(new_edge_loop, get_edgeloop_length(new_edge_loop)));
                loop_travel_for_split(edge_loop[pos1]->vert, new_edge_loop);
                if (is_closed_edge_loop(new_edge_loop))
                    edge_loop_queue.push(std::make_pair(new_edge_loop, get_edgeloop_length(new_edge_loop)));
                break;
            }
            else if (count_corners.size() == 0)
            {
                // search all edgeloop emitting from the edge_loop vertices, find the edgeloop with max length to split
                std::vector<std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real>> emitting_edge_loops_with_length;
                for (auto e : edge_loop)
                {
                    auto v = e->pair->vert;
                    auto edge = v->edge;
                    do
                    {
                        if (!complex_edge_tag[edge->id] && !edge->tag && !edge->pair->tag)
                        {
                            std::vector<MeshLib::HE_edge<Real> *> new_edge_loop;
                            edge_loop_travel(edge, new_edge_loop, singular_vertex_tag, complex_edge_tag, false);
                            Real length = get_edgeloop_length(new_edge_loop);
                            emitting_edge_loops_with_length.emplace_back(std::make_pair(new_edge_loop, length));
                            for (auto &e2 : new_edge_loop)
                            {
                                e2->tag = e2->pair->tag = false;
                            }
                        }
                        edge = edge->pair->next;
                    } while (edge != v->edge);
                }
                std::sort(emitting_edge_loops_with_length.begin(), emitting_edge_loops_with_length.end(),
                          [](const std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real> &a, const std::pair<std::vector<MeshLib::HE_edge<Real> *>, Real> &b)
                          {
                              return a.second > b.second;
                          });
                tag_edge_loop_for_split(emitting_edge_loops_with_length.front().first, true);
            }
            else
                break;
        }

        // loop_split_num = 2;
    }
}
////////////////////////////////////////////////
template <typename Real>
bool BaseComplex<Real>::is_complex_corner(MeshLib::HE_vert<Real> *hv, bool singular_tag)
{
    if (singular_tag)
        return true;
    auto he = hv->edge;
    int count = 0;
    do
    {
        if (complex_edge_tag[he->id])
            count++;
        he = he->pair->next;
    } while (he != hv->edge);
    return count > 2;
}
//////////////////////////////////////////////
template <typename Real>
bool BaseComplex<Real>::is_complex_vertex(MeshLib::HE_vert<Real> *hv, bool singular_tag)
{
    if (singular_tag)
        return true;
    auto he = hv->edge;
    int count = 0;
    do
    {
        if (complex_edge_tag[he->id])
            return true;
        he = he->pair->next;
    } while (he != hv->edge);
    return false;
}
////////////////////////////////////////////////

/////////////////////////////////////////////
template class BaseComplex<double>;