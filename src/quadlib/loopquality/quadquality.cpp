
#include "quadquality.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <queue>
#include <unordered_set>
#include "GraphColoring.h"
#include "looputil.h"
#include "looptravel.h"
#include "MeshWriter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif

template <typename Real>
QuadQuality<Real>::QuadQuality(MeshLib::Mesh3D<Real> *mesh, bool _verbose)
    : quad_mesh(mesh), verbose(_verbose)
{
    if (quad_mesh == nullptr || quad_mesh->is_quad() == false)
    {
        std::cerr << "QuadQuality: input mesh is nullptr or is not pure quad!\n";
        exit(0);
    }

    boundary_vertex_tag.assign(quad_mesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto vert = quad_mesh->get_vertex(i);
        if (quad_mesh->is_on_boundary(vert))
            boundary_vertex_tag[i] = true;
    }

    face_areas.assign(quad_mesh->get_num_of_faces(), 0);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        face_areas[i] = quad_mesh->get_face(i)->GetArea();
    }

    if (verbose)
    {
        std::cout << "QuadQuality: input mesh has " << quad_mesh->get_num_of_vertices() << " vertices, " << quad_mesh->get_num_of_faces() << " quad faces.\n";
        std::cout << "QuadQuality: input mesh has " << quad_mesh->get_num_of_boundaries() << " disjointed boundaries, " << quad_mesh->get_num_of_components() << " disjointed components.\n";
    }
    compute_quad_quality();
    compute_mesh_quality();
    compute_loop_and_layout_quality();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_base_complex_edges_as_obj(const char filename[])
{
    std::ofstream objfile(filename);
    std::unordered_map<ptrdiff_t, size_t> vertex_map;
    size_t vertex_id = 1;
    for (size_t i = 0; i < complex_edge_tag.size(); ++i)
    {
        if (complex_edge_tag[i])
        {
            auto edge = quad_mesh->get_edge(i);
            if (vertex_map.find(edge->pair->vert->id) == vertex_map.end())
            {
                vertex_map[edge->pair->vert->id] = vertex_id++;
                objfile << "v " << edge->pair->vert->pos << '\n';
            }
            if (vertex_map.find(edge->vert->id) == vertex_map.end())
            {
                vertex_map[edge->vert->id] = vertex_id++;
                objfile << "v " << edge->vert->pos << '\n';
            }
        }
    }
    for (size_t i = 0; i < complex_edge_tag.size(); ++i)
    {
        if (complex_edge_tag[i])
        {
            auto edge = quad_mesh->get_edge(i);
            objfile << "l " << vertex_map[edge->pair->vert->id] << ' ' << vertex_map[edge->vert->id] << '\n';
        }
    }
    objfile.close();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_base_complex_faces_as_off(const char filename[])
{
    // build graph
    auto num_complex = *std::max_element(face_complex_ids.begin(), face_complex_ids.end()) + 1;
    std::vector<std::unordered_set<size_t>> graph_edges(num_complex);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge->face == nullptr || edge->pair->face == nullptr)
            continue;
        auto cid0 = face_complex_ids[edge->face->id];
        auto cid1 = face_complex_ids[edge->pair->face->id];
        if (cid0 != cid1)
        {
            graph_edges[cid0].insert(cid1);
            graph_edges[cid1].insert(cid0);
        }
    }
    std::vector<std::vector<size_t>> colored_vertices;
    greedy_graph_coloring(num_complex, graph_edges, colored_vertices);

    std::vector<int> rand_color(3 * colored_vertices.size());
    for (auto &color : rand_color)
    {
        color = rand() % 256;
    }
    std::vector<size_t> complex_color(face_complex_ids.size());
    for (size_t i = 0; i < colored_vertices.size(); ++i)
    {
        for (const auto vertex : colored_vertices[i])
        {
            complex_color[vertex] = i;
        }
    }
    std::ofstream offfile(filename);

    offfile << "COFF\n";
    offfile << quad_mesh->get_num_of_vertices() << ' ' << quad_mesh->get_num_of_faces() << ' ' << 0 << '\n';

    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
        offfile << quad_mesh->get_vertex(i)->pos << '\n';
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        auto he = face->edge;
        offfile << face->valence;
        do
        {
            offfile << ' ' << he->pair->vert->id;
            he = he->next;
        } while (he != face->edge);
        auto cid = 3 * complex_color[face_complex_ids[i]];
        offfile << ' ' << rand_color[cid] << ' ' << rand_color[cid + 1] << ' ' << rand_color[cid + 2] << '\n';
    }

    offfile.close();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_base_complex_edges_as_ply(const char filename[])
{
    SaveMarkedEdge_as_ply(quad_mesh, filename, complex_edge_tag, 0.0f);
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_base_complex_faces_as_ply(const char filename[])
{
    SavePLYmesh_with_float_storage(quad_mesh, filename, &face_complex_ids, false);
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_faceloop_quality(const char filename[])
{
    std::ofstream csvfile(filename);
    csvfile << "NumF,AreaLoop,Closeness,SelfIntersection,RotationIndex\n";
    for (size_t i = 0; i < face_loop_collection.size(); ++i)
    {
        csvfile << face_loop_collection[i].size() << ',';
        Real total_area = 0;
        for (const auto *face : face_loop_collection[i])
        {
            total_area += face->GetArea();
        }
        csvfile << total_area << ',';
        csvfile << closeness_face_loops[i] << ',' << self_intersection_face_loops[i] << ',' << total_curvature_face_loops[i] << '\n';
    }
    csvfile.close();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_edgeloop_quality(const char filename[])
{
    std::ofstream csvfile(filename);
    csvfile << "NumE,LengthLoop,Closeness,SelfIntersection,RotationIndex\n";
    for (size_t i = 0; i < edge_loop_collection.size(); ++i)
    {
        csvfile << edge_loop_collection[i].size() << ',';
        Real total_length = 0;
        for (const auto *edge : edge_loop_collection[i])
        {
            total_length += edge->GetLength();
        }
        csvfile << total_length << ',';
        csvfile << closeness_edge_loops[i] << ',' << self_intersection_edge_loops[i] << ',' << total_curvature_edge_loops[i] << '\n';
    }
    csvfile.close();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_quadface_quality(const char filename[])
{
    std::ofstream csvfile(filename);
    csvfile << "Planarity,Regularity\n";
    for (size_t i = 0; i < planarity.size(); ++i)
    {
        csvfile << planarity[i] << ',' << regularity[i] << '\n';
    }
    csvfile.close();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_mesh_quality(const char filename[])
{
    std::ofstream csvfile(filename);
    csvfile << "NumV,NumQF,NumIrregularVertices,NumComplexes,NumEdgeLoops,NumFaceLoops,Planarity,Regularity,WEdgeLoopInd,WFaceLoopInd,WEdgeSI,WFaceSI,SimpleEdgeLoopLengthRatio,SimpleFaceLoopAreaRatio\n";
    csvfile << quad_mesh->get_num_of_vertices() << ',';
    csvfile << quad_mesh->get_num_of_faces() << ',';
    csvfile << num_irregular_vertices << ',';
    csvfile << *std::max_element(face_complex_ids.begin(), face_complex_ids.end()) + 1 << ',';
    csvfile << closeness_edge_loops.size() << ',' << closeness_face_loops.size() << ',';
    csvfile << std::accumulate(planarity.begin(), planarity.end(), Real{0}) / planarity.size() << ',';
    csvfile << std::accumulate(regularity.begin(), regularity.end(), Real{0}) / regularity.size() << ',';

    Real w_edge_loop_ind = 0, w_face_loop_ind = 0;
    Real total_edgeloop_length = 0, total_faceloop_area = 0;
    Real w_edge_SI = 0, w_face_SI = 0;
    for (size_t i = 0; i < edge_loop_collection.size(); ++i)
    {
        Real edge_length = 0;
        for (const auto *edge : edge_loop_collection[i])
        {
            edge_length += edge->GetLength();
        }
        total_edgeloop_length += edge_length;
        w_edge_loop_ind += edge_length * total_curvature_edge_loops[i];

        w_edge_SI += edge_length * self_intersection_edge_loops[i];
    }
    w_edge_loop_ind /= total_edgeloop_length;
    w_edge_SI /= total_edgeloop_length;

    for (size_t i = 0; i < face_loop_collection.size(); ++i)
    {
        Real face_area = 0;
        for (const auto *face : face_loop_collection[i])
        {
            face_area += face->GetArea();
        }
        total_faceloop_area += face_area;
        w_face_loop_ind += face_area * total_curvature_face_loops[i];

        w_face_SI += face_area * self_intersection_face_loops[i];
    }
    w_face_loop_ind /= total_faceloop_area;
    w_face_SI /= total_faceloop_area;

    Real sedge_ratio = get_simple_edgeloop_ratio();
    Real sarea_ratio = get_simple_faceloop_ratio();

    csvfile << w_edge_loop_ind << ',' << w_face_loop_ind << ',' << w_edge_SI << ',' << w_face_SI << ',' << sedge_ratio << ',' << sarea_ratio << '\n';

    csvfile.close();
    if (verbose)
    {
        std::cout << "Weighted edge loop ind: " << w_edge_loop_ind << '\n';
        std::cout << "Weighted face loop ind: " << w_face_loop_ind << '\n';
        std::cout << "Weighted edge SI: " << w_edge_SI << '\n';
        std::cout << "Weighted face SI: " << w_face_SI << '\n';
        std::cout << "Simple edge loop length ratio: " << sedge_ratio << '\n';
        std::cout << "Simple face loop area ratio: " << sarea_ratio << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_simple_faceloop_ratio()
{
    Real sarea = 0, total_area = 0;
    quad_mesh->reset_faces_tag(false);
    for (size_t i = 0; i < face_loop_collection.size(); ++i)
    {
        Real temp_area = 0;
        for (const auto *face : face_loop_collection[i])
        {
            temp_area += face_areas[face->id];
        }
        if (self_intersection_face_loops[i] == 0 && total_curvature_face_loops[i] <= 1.1)
            sarea += temp_area;
        total_area += temp_area;
    }
    if (total_area == 0)
        return 1;
    return sarea / total_area;
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_simple_edgeloop_ratio()
{
    Real s_weight = 0, total_weight = 0;
    for (size_t i = 0; i < edge_loop_collection.size(); ++i)
    {
        Real tmp_weight = 0;
        for (const auto *e : edge_loop_collection[i])
        {
            Real weight = 0;
            if (e->face)
                weight += face_areas[e->face->id];
            if (e->pair->face)
                weight += face_areas[e->pair->face->id];
            tmp_weight += weight;
        }
        if (self_intersection_edge_loops[i] == 0 && total_curvature_edge_loops[i] <= 1.1)
            s_weight += tmp_weight;

        total_weight += tmp_weight;
    }
    if (total_weight == 0)
        return 1;
    return s_weight / total_weight;
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_simple_faceloop_ratio_new()
{
    Real sarea = 0, total_area = 0;
    for (size_t i = 0; i < face_loop_collection.size(); ++i)
    {
        Real temp_area = 0;
        for (const auto *face : face_loop_collection[i])
        {
            temp_area += face_areas[face->id];
        }
        if (self_intersection_face_loops[i] == 0)
            sarea += temp_area;
        total_area += temp_area;
    }
    if (total_area == 0)
        return 1;
    return sarea / total_area;
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_simple_edgeloop_ratio_new()
{
    Real s_weight = 0, total_weight = 0;
    for (size_t i = 0; i < edge_loop_collection.size(); ++i)
    {
        Real tmp_weight = 0;
        for (const auto *e : edge_loop_collection[i])
        {
            Real weight = 0;
            if (e->face)
                weight += face_areas[e->face->id];
            if (e->pair->face)
                weight += face_areas[e->pair->face->id];
            tmp_weight += weight;
        }
        if (self_intersection_edge_loops[i] == 0)
            s_weight += tmp_weight;

        total_weight += tmp_weight;
    }
    if (total_weight == 0)
        return 1;
    return s_weight / total_weight;
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_faceloop_spriality_ratio()
{
    Real sarea = 0, total_area = 0;
    for (size_t i = 0; i < face_loop_collection.size(); ++i)
    {
        Real temp_area = 0;
        for (const auto *face : face_loop_collection[i])
        {
            temp_area += face_areas[face->id];
        }
        sarea += temp_area * (total_curvature_face_loops[i] >= 1.1 ? 1 : 0);
        total_area += temp_area;
    }
    if (total_area == 0)
        return 1;
    return sarea / total_area;
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_edgeloop_spriality_ratio()
{
    Real s_weight = 0, total_weight = 0;
    for (size_t i = 0; i < edge_loop_collection.size(); ++i)
    {
        Real tmp_weight = 0;
        for (const auto *e : edge_loop_collection[i])
        {
            Real weight = 0;
            if (e->face)
                weight += face_areas[e->face->id];
            if (e->pair->face)
                weight += face_areas[e->pair->face->id];
            tmp_weight += weight;
        }
        s_weight += tmp_weight * (total_curvature_edge_loops[i] >= 1.1 ? 1 : 0);
        total_weight += tmp_weight;
    }
    if (total_weight == 0)
        return 1;
    return s_weight / total_weight;
}
////////////////////////////////////////////////
template <typename Real>
size_t QuadQuality<Real>::get_num_of_complex()
{
    return *std::max_element(face_complex_ids.begin(), face_complex_ids.end()) + 1;
}
////////////////////////////////////////////////
template <typename Real>
bool QuadQuality<Real>::is_checkerable()
{
    for (size_t i = 0; i < face_loop_collection.size(); ++i)
    {
        if (face_loop_collection[i].size() % 2 != 0)
            return false;
    }
    for (size_t i = 0; i < edge_loop_collection.size(); ++i)
    {
        if (edge_loop_collection[i].size() % 2 != 0 || edge_loop_collection[i].size() == 2)
            return false;
    }
    return true;
}
////////////////////////////////////////////////
template <typename Real>
size_t QuadQuality<Real>::get_irregular_vertex_num()
{
    return num_irregular_vertices;
}
////////////////////////////////////////////////
template <typename Real>
const std::vector<ptrdiff_t> &QuadQuality<Real>::get_face_complex_ids() const
{
    return face_complex_ids;
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::get_complex_distribution(Real &max_area_ratio, Real &min_area_ratio, Real &mean_area_ratio, Real &min_edge_length)
{
    size_t num_complex = get_num_of_complex();
    std::vector<Real> area_distribution(num_complex, 0);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        auto cid = face_complex_ids[i];
        area_distribution[cid] += face->GetArea();
    }

    Real total_area = std::accumulate(area_distribution.begin(), area_distribution.end(), Real{0});
    max_area_ratio = *std::max_element(area_distribution.begin(), area_distribution.end()) / total_area;
    min_area_ratio = *std::min_element(area_distribution.begin(), area_distribution.end()) / total_area;
    mean_area_ratio = total_area / area_distribution.size();
    // std::cout << "Complex area distribution: max = " << max_area_ratio << ", min = " << min_area_ratio << ", mean = " << mean_area_ratio << std::endl;
    std::unordered_map<MySortedTuple<ptrdiff_t, 2, false>, Real> edge_distribution;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge > edge->pair)
            continue;
        auto cid0 = edge->face ? face_complex_ids[edge->face->id] : -1;
        auto cid1 = edge->pair->face ? face_complex_ids[edge->pair->face->id] : -1;
        if (cid0 != cid1)
        {
            auto key = MySortedTuple<ptrdiff_t, 2, false>(std::min(cid0, cid1), std::max(cid0, cid1));
            edge_distribution[key] += edge->GetLength();
        }
    }
    min_edge_length = std::numeric_limits<Real>::max();
    for (const auto &pair : edge_distribution)
    {
        if (pair.second < min_edge_length)
            min_edge_length = pair.second;
    }
    // std::cout << "Complex edge distribution: min edge length = " << min_edge_length << std::endl;
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::compute_quad_quality()
{
    planarity.assign(quad_mesh->get_num_of_faces(), 0);
    regularity.assign(quad_mesh->get_num_of_faces(), 0);
    Real angle[4], angle_diff[4];

    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        auto edge = face->edge;
        auto v0 = edge->vert;
        auto v1 = edge->next->vert;
        auto v2 = edge->next->next->vert;
        auto v3 = edge->pair->vert;

        angle[0] = compute_angle(v3->pos, v0->pos, v1->pos);
        angle[1] = compute_angle(v0->pos, v1->pos, v2->pos);
        angle[2] = compute_angle(v1->pos, v2->pos, v3->pos);
        angle[3] = compute_angle(v2->pos, v3->pos, v0->pos);
        for (int j = 0; j < 4; j++)
            angle_diff[j] = fabs(angle[j] - 90);

        planarity[i] = fabs(angle[0] + angle[1] + angle[2] + angle[3] - 360);
        regularity[i] = (angle_diff[0] + angle_diff[1] + angle_diff[2] + angle_diff[3]) / 4;
    }
    Real avg_p, max_p, var_p;
    Real avg_r, max_r, var_r;
    compute_statistics(planarity, avg_p, max_p, var_p);
    compute_statistics(regularity, avg_r, max_r, var_r);
    if (verbose)
    {
        std::cout << "Planarity (in degree): " << avg_p << " " << max_p << " " << var_p << '\n';
        std::cout << "Regularity (in degree): " << avg_r << " " << max_r << " " << var_r << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::compute_mesh_quality()
{
    num_irregular_vertices = 0;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto vert = quad_mesh->get_vertex(i);
        if (boundary_vertex_tag[i])
        {
            if (vert->degree != 3)
                num_irregular_vertices++;
        }
        else
        {
            if (vert->degree != 4)
                num_irregular_vertices++;
        }
    }
    if (verbose)
    {
        std::cout << "Irregular vertices: " << num_irregular_vertices << '\n';
        std::cout << "Irregular vertex ratio: " << num_irregular_vertices / static_cast<Real>(quad_mesh->get_num_of_vertices()) << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
Real QuadQuality<Real>::get_mean_scaled_jacobian()
{
    Real total_sj = 0;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        auto edge = face->edge;
        Real min_sj = std::numeric_limits<Real>::max();
        do
        {
            auto e1 = edge->pair->vert->pos - edge->vert->pos;
            auto e2 = edge->next->vert->pos - edge->vert->pos;
            Real sj = (e1.Cross(e2)).Length() / (e1.Length() * e2.Length() + static_cast<Real>(1e-12));
            if (sj < min_sj)
                min_sj = sj;
            edge = edge->next;
        } while (edge != face->edge);
        total_sj += min_sj;
    }
    return total_sj / quad_mesh->get_num_of_faces();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::compute_loop_and_layout_quality()
{

    std::vector<bool> singular_vertex_tag(quad_mesh->get_num_of_vertices(), false), boundary_vertex_tag(quad_mesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = quad_mesh->get_vertex(i);
        boundary_vertex_tag[i] = quad_mesh->is_on_boundary(hv);
        singular_vertex_tag[i] = boundary_vertex_tag[i] ? hv->degree != 3 : hv->degree != 4;
    }

    quad_mesh->reset_edges_tag(false);

    complex_edge_tag.assign(quad_mesh->get_num_of_edges(), false);
    // compare the base complex
    ptrdiff_t num_edge_loops = 0;
    ptrdiff_t num_base_complex_edge_loops = 0;

    closeness_edge_loops.clear(), self_intersection_edge_loops.clear(), total_curvature_edge_loops.clear();
    edge_loop_collection.clear();
    closeness_edge_loops.reserve(quad_mesh->get_num_of_edges());
    self_intersection_edge_loops.reserve(quad_mesh->get_num_of_edges());
    total_curvature_edge_loops.reserve(quad_mesh->get_num_of_edges());
    edge_loop_collection.reserve(quad_mesh->get_num_of_edges());
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = quad_mesh->get_vertex(i);
        if (singular_vertex_tag[i])
        {
            auto he = hv->edge;
            do
            {
                if (he->tag || he->pair->tag || he->face == nullptr)
                    ;
                else
                {
                    num_edge_loops++;
                    std::vector<MeshLib::HE_edge<Real> *> edge_loop;
                    edge_loop_travel(he, edge_loop, singular_vertex_tag, complex_edge_tag, true);
                    edge_loop_collection.emplace_back(edge_loop);
                    bool closed;
                    int num_self_intersection;
                    Real total_curvature;
                    edge_loop_quality(edge_loop, closed, num_self_intersection, total_curvature);
                    closeness_edge_loops.emplace_back(closed ? 1 : 0);
                    self_intersection_edge_loops.emplace_back(num_self_intersection);
                    total_curvature_edge_loops.emplace_back(total_curvature);
                }
                he = he->pair->next;
            } while (he != hv->edge);
        }
    }

    num_base_complex_edge_loops = num_edge_loops;

    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge->tag || edge->pair->tag || edge->face == nullptr)
            continue;
        num_edge_loops++;
        std::vector<MeshLib::HE_edge<Real> *> edge_loop;
        edge_loop_travel(edge, edge_loop, singular_vertex_tag, complex_edge_tag, false);
        edge_loop_collection.emplace_back(edge_loop);
        bool closed;
        int num_self_intersection;
        Real total_curvature;
        edge_loop_quality(edge_loop, closed, num_self_intersection, total_curvature);
        closeness_edge_loops.emplace_back(closed ? 1 : 0);
        self_intersection_edge_loops.emplace_back(num_self_intersection);
        total_curvature_edge_loops.emplace_back(total_curvature);
    }
    // dump edgeloops
    // for (size_t ei = 0; ei < edge_loop_collection.size(); ei++)
    // {
    // 	char filename[256];
    // 	sprintf_s(filename, "edge_loop_%d.obj", (int)ei);
    // 	export_edge_loop_as_obj(edge_loop_collection[ei], filename);
    // }
    if (verbose)
    {
        std::cout << "Number of edge loops: " << num_edge_loops << '\n';
        std::cout << "Number of base complex edge loops: " << num_base_complex_edge_loops << '\n';
    }
    Real avg_si, max_si, var_si;
    compute_statistics(self_intersection_edge_loops, avg_si, max_si, var_si);
    if (verbose)
        std::cout << "Self-intersection of edge loops: " << avg_si << " " << max_si << " " << var_si << '\n';
    Real avg_closeness, max_closeness, var_closeness;
    compute_statistics(closeness_edge_loops, avg_closeness, max_closeness, var_closeness);
    if (verbose)
        std::cout << "Closeness of edge loops: " << avg_closeness << " " << max_closeness << " " << var_closeness << '\n';
    Real avg_curvature, max_curvature, var_curvature;
    compute_statistics(total_curvature_edge_loops, avg_curvature, max_curvature, var_curvature);
    if (verbose)
        std::cout << "Rotation index of edge loops: " << avg_curvature << " " << max_curvature << " " << var_curvature << '\n';

    // compute the number of complex
    quad_mesh->reset_faces_tag(false);
    size_t num_complex = 0;
    face_complex_ids.assign(quad_mesh->get_num_of_faces(), 0);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        if (face->tag)
            continue;
        std::queue<MeshLib::HE_face<Real> *> face_queue;
        face_queue.push(face);
        face->tag = true;
        while (!face_queue.empty())
        {
            auto f = face_queue.front();
            face_queue.pop();
            face_complex_ids[f->id] = num_complex;
            auto e = f->edge;
            do
            {
                if (complex_edge_tag[e->id] == false && complex_edge_tag[e->pair->id] == false && e->pair->face && e->pair->face->tag == false)
                {
                    e->pair->face->tag = true;
                    face_queue.push(e->pair->face);
                }
                e = e->next;
            } while (e != f->edge);
        }
        num_complex++;
    }
    if (verbose)
        std::cout << "Number of complexes: " << num_complex << '\n';

    // compute face loops
    closeness_face_loops.clear(), self_intersection_face_loops.clear(), total_curvature_face_loops.clear();
    // std::vector<std::vector<MeshLib::HE_face<Real> *>> face_loop_collection;
    face_loop_collection.clear();
    closeness_face_loops.reserve(quad_mesh->get_num_of_edges());
    self_intersection_face_loops.reserve(quad_mesh->get_num_of_edges());
    total_curvature_face_loops.reserve(quad_mesh->get_num_of_edges());
    face_loop_collection.reserve(quad_mesh->get_num_of_edges());
    quad_mesh->reset_edges_tag(false);
    ptrdiff_t num_face_loops = 0;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto edge = quad_mesh->get_edge(i);
        if (edge->face == nullptr || edge->tag || edge->pair->tag)
            continue;

        std::vector<MeshLib::HE_face<Real> *> face_loop;
        face_loop_travel(edge, face_loop);
        if (face_loop.size() < 2)
            continue;
        bool closed;
        int num_self_intersection;
        Real total_curvature;
        face_loop_quality(face_loop, closed, num_self_intersection, total_curvature);
        closeness_face_loops.emplace_back(closed ? 1 : 0);
        self_intersection_face_loops.emplace_back(num_self_intersection);
        total_curvature_face_loops.emplace_back(total_curvature);
        face_loop_collection.emplace_back(face_loop);
        num_face_loops++;
    }
    // dump faceloops
    // for (size_t fi = 0; fi < face_loop_collection.size(); fi++)
    // {
    // 	char filename[256];
    // 	sprintf_s(filename, "face_loop_%d.obj", (int)fi);
    // 	export_face_loop_as_obj(face_loop_collection[fi], filename);
    // }
    if (verbose)
        std::cout << "Number of face loops: " << num_face_loops << '\n';
    Real avg_si_f, max_si_f, var_si_f;
    compute_statistics(self_intersection_face_loops, avg_si_f, max_si_f, var_si_f);
    if (verbose)
        std::cout << "Self-intersection of face loops: " << avg_si_f << " " << max_si_f << " " << var_si_f << '\n';
    Real avg_closeness_f, max_closeness_f, var_closeness_f;
    compute_statistics(closeness_face_loops, avg_closeness_f, max_closeness_f, var_closeness_f);
    if (verbose)
        std::cout << "Closeness of face loops: " << avg_closeness_f << " " << max_closeness_f << " " << var_closeness_f << '\n';
    Real avg_curvature_f, max_curvature_f, var_curvature_f;
    compute_statistics(total_curvature_face_loops, avg_curvature_f, max_curvature_f, var_curvature_f);
    if (verbose)
        std::cout << "Rotation index of face loops: " << avg_curvature_f << " " << max_curvature_f << " " << var_curvature_f << '\n';
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::edge_loop_quality(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, bool &closed, int &num_self_intersection, Real &rotation_index)
{
    std::unordered_map<ptrdiff_t, int> vertex_count;
    for (const auto *edge : edge_loop)
    {
        vertex_count[edge->vert->id]++;
        vertex_count[edge->pair->vert->id]++;
    }
    closed = edge_loop.front()->pair->vert == edge_loop.back()->vert;
    num_self_intersection = 0;
    for (const auto &v : vertex_count)
    {
        if (v.second > 2)
        {
            num_self_intersection++;
        }
    }

    if (edge_loop.size() < 2)
    {
        rotation_index = 0;
        return;
    }

    // compute the rotation index of edge_loop
    std::vector<TinyVector<Real, 3>> points;
    points.reserve(edge_loop.size() + 1);
    for (const auto *edge : edge_loop)
    {
        points.emplace_back(edge->pair->vert->pos);
    }
    if (!closed)
        points.emplace_back(edge_loop.back()->vert->pos);

    compute_rotational_index(points, closed, rotation_index);
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::compute_rotational_index(const std::vector<TinyVector<Real, 3>> &points, bool closed, Real &rotation_index)
{
    TinyVector<Real, 3> line_direction, line_origin;
    FittingPlane<Real>(points, closed, line_direction, line_origin);
    std::vector<TinyVector<Real, 2>> line_points_2d(points.size());
    TinyVector<Real, 3> rand_dir(static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX), static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX), static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX));
    TinyVector<Real, 3> Xdir = line_direction.UnitCross(rand_dir);
    TinyVector<Real, 3> Ydir = line_direction.UnitCross(Xdir);
    for (size_t i = 0; i < points.size(); ++i)
    {
        auto p = points[i] - line_origin;
        line_points_2d[i][0] = p.Dot(Xdir);
        line_points_2d[i][1] = p.Dot(Ydir);
    }
    rotation_index = 0;
    // size_t edge_loop_size = closed ? points.size() : points.size() - 1;
    // size_t end = closed ? edge_loop_size : edge_loop_size - 2;
    size_t end = closed ? points.size() : points.size() - 2;
    for (size_t i = 0; i < end; ++i)
    {
        auto v1 = line_points_2d[(i + 1) % line_points_2d.size()];
        auto v0 = line_points_2d[i];
        auto v2 = line_points_2d[(i + 2) % line_points_2d.size()];
        auto v10 = v1 - v0, v21 = v2 - v1;
        v10.Normalize();
        v21.Normalize();
        Real angle = std::acos(std::min(static_cast<Real>(1.0), std::max(static_cast<Real>(-1.0), v10.Dot(v21))));
        Real cross = v10[0] * v21[1] - v10[1] * v21[0];
        if (cross < 0)
            angle = -angle;
        rotation_index += angle;
    }
    rotation_index = fabs(rotation_index / (2 * M_PI));
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_edge_loop_as_obj(const std::vector<MeshLib::HE_edge<Real> *> &edge_loop, const char filename[])
{
    std::ofstream objfile(filename);
    std::unordered_map<ptrdiff_t, size_t> vertex_map;
    size_t vertex_id = 1;
    for (const auto *edge : edge_loop)
    {
        if (vertex_map.find(edge->pair->vert->id) == vertex_map.end())
        {
            vertex_map[edge->pair->vert->id] = vertex_id++;
            objfile << "v " << edge->pair->vert->pos << '\n';
        }
        if (vertex_map.find(edge->vert->id) == vertex_map.end())
        {
            vertex_map[edge->vert->id] = vertex_id++;
            objfile << "v " << edge->vert->pos << '\n';
        }
    }
    for (const auto *edge : edge_loop)
    {
        objfile << "l " << vertex_map[edge->pair->vert->id] << ' ' << vertex_map[edge->vert->id] << '\n';
    }
    objfile.close();
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::face_loop_quality(const std::vector<MeshLib::HE_face<Real> *> &face_loop, bool &closed, int &num_self_intersection, Real &rotation_index)
{
    auto front_f = face_loop.front();
    auto edge = front_f->edge;
    closed = false;
    do
    {
        if (edge->pair->face == face_loop.back())
        {
            closed = true;
            break;
        }
        edge = edge->next;
    } while (edge != front_f->edge);

    std::unordered_map<ptrdiff_t, int> face_count;
    for (const auto *face : face_loop)
    {
        face_count[face->id]++;
    }
    num_self_intersection = 0;
    for (const auto &v : face_count)
    {
        if (v.second > 1)
        {
            num_self_intersection++;
        }
    }

    if (face_loop.size() < 3)
    {
        rotation_index = 0;
        return;
    }

    // compute the rotation index of face_loop
    std::vector<TinyVector<Real, 3>> points;
    points.reserve(face_loop.size());
    for (const auto *face : face_loop)
    {
        points.emplace_back(face->GetCentroid());
    }
    compute_rotational_index(points, closed, rotation_index);
}
////////////////////////////////////////////////
template <typename Real>
void QuadQuality<Real>::export_face_loop_as_obj(const std::vector<MeshLib::HE_face<Real> *> &face_loop, const char filename[])
{
    std::ofstream objfile(filename);
    std::unordered_map<ptrdiff_t, size_t> vertex_map;
    size_t vertex_id = 1;
    for (const auto *face : face_loop)
    {
        auto edge = face->edge;
        do
        {
            if (vertex_map.find(edge->pair->vert->id) == vertex_map.end())
            {
                vertex_map[edge->pair->vert->id] = vertex_id++;
                objfile << "v " << edge->pair->vert->pos << '\n';
            }
            edge = edge->next;
        } while (edge != face->edge);
    }
    for (const auto *face : face_loop)
    {
        auto edge = face->edge;
        objfile << "f";
        do
        {
            objfile << ' ' << vertex_map[edge->pair->vert->id];
            edge = edge->next;
        } while (edge != face->edge);
        objfile << '\n';
    }
    objfile.close();
}
////////////////////////////////////////////////
template class QuadQuality<double>;