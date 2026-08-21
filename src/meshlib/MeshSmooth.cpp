#include "MeshSmooth.h"
#include <omp.h>

////////////////////////////////////////////////////////////////////////////////////////
template <typename Real>
void mesh_laplacian_smoothing(MeshLib::Mesh3D<Real> *mesh, int iterations, Real taubin_weight)
{
    if (!mesh || iterations <= 0)
        return;
    std::vector<TinyVector<Real, 3>> new_positions(mesh->get_num_of_vertices());
    for (int iter = 0; iter < iterations; iter++)
    {
#pragma omp parallel for schedule(dynamic)
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
        {
            auto vert = mesh->get_vertex(i);
            bool is_on_boundary = mesh->is_on_boundary(vert);
            if (vert->degree == 2 && is_on_boundary)
            {
                new_positions[i] = vert->pos;
                continue;
            }
            auto edge = vert->edge;
            TinyVector<Real, 3> feature_center, center;
            int feature_count = 0, count = 0;
            Real total_weight = 0, feature_weight = 0;
            do
            {
                auto weight = 1 / ((vert->pos - edge->vert->pos).SquaredLength() + Real(1.0e-12));

                if (edge->tag || mesh->is_on_boundary(edge))
                {
                    // feature_center += edge->vert->pos;
                    // feature_count++;
                    feature_center += weight * edge->vert->pos;
                    feature_weight += weight;
                }
                // center += edge->vert->pos;
                // count++;

                center += weight * edge->vert->pos;
                total_weight += weight;

                edge = edge->pair->next;
            } while (edge != vert->edge);

            if (feature_count == 2 && vert->tag == false)
                // new_positions[i] = (1 - taubin_weight) * vert->pos + (taubin_weight / feature_count) * feature_center;
                new_positions[i] = (1 - taubin_weight) * vert->pos + (taubin_weight / feature_weight) * feature_center;
            else if (feature_count > 0)
                new_positions[i] = vert->pos; // corner vertex
            else
                // new_positions[i] = (1 - taubin_weight) * vert->pos + (taubin_weight / count) * center;
                new_positions[i] = (1 - taubin_weight) * vert->pos + (taubin_weight / total_weight) * center;
        }
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
        {
            mesh->get_vertex(i)->pos = new_positions[i];
        }
    }
} ////////////////////////////////////////////////////////////////////////////////////////
template <typename Real>
void mesh_winslow_smoothing(MeshLib::Mesh3D<Real> *mesh, int iterations, Real taubin_weight)
{
    if (!mesh || iterations <= 0)
        return;
    std::vector<TinyVector<Real, 3>> new_positions(mesh->get_num_of_vertices());
    std::vector<std::vector<ptrdiff_t>> vertex_one_rings(mesh->get_num_of_vertices());
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(mesh->get_num_of_vertices()); i++)
    {
        auto vert = mesh->get_vertex(i);
        auto he = vert->edge;
        vertex_one_rings[i].reserve(vert->degree);
        bool is_on_boundary = mesh->is_on_boundary(vert);
        std::vector<ptrdiff_t> feature_neighbors;
        do
        {
            if (!is_on_boundary)
                vertex_one_rings[i].emplace_back(he->vert->id);
            else if (mesh->is_on_boundary(he))
            {
                vertex_one_rings[i].emplace_back(he->vert->id);
            }
            if (he->tag)
            {
                feature_neighbors.emplace_back(he->vert->id);
            }
            he = he->pair->next;
        } while (he != vert->edge);
        if (feature_neighbors.size() == size_t{2} && vert->tag == false)
        {
            vertex_one_rings[i] = feature_neighbors;
        }
        else if (!feature_neighbors.empty())
        {
            vertex_one_rings[i].clear();
        }
    }
    for (int iter = 0; iter < iterations; iter++)
    {
#pragma omp parallel for schedule(dynamic)
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
        {
            auto vert = mesh->get_vertex(i);
            if ((vert->degree == 2 && mesh->is_on_boundary(vert)) || vertex_one_rings[i].empty())
            {
                new_positions[i] = vert->pos;
                continue;
            }

            auto he = vert->edge;
            TinyVector<Real, 3> center(0, 0, 0);
            const auto &one_ring = vertex_one_rings[i];
            Real weight = 0, total_weight = 0;
            if (one_ring.size() % 2 != 0)
            {
                for (size_t j = 0; j < one_ring.size(); ++j)
                {
                    weight = 1;
                    center += mesh->get_vertex(one_ring[j])->pos;
                    total_weight += weight;
                }
            }
            else if (one_ring.size() == size_t{2})
            {
                center = (mesh->get_vertex(one_ring[0])->pos + mesh->get_vertex(one_ring[1])->pos);
                total_weight = 2;
            }
            else
            {
                for (size_t j = 0, half = one_ring.size() / 2; j < half; ++j)
                {
                    weight = 1 / ((mesh->get_vertex(one_ring[j])->pos - mesh->get_vertex(one_ring[(j + half) % one_ring.size()])->pos).SquaredLength() + Real(1.0e-12));
                    center += (weight / 2) * (mesh->get_vertex(one_ring[j])->pos + mesh->get_vertex(one_ring[(j + half) % one_ring.size()])->pos);
                    total_weight += weight;
                }
            }
            new_positions[i] = (1 - taubin_weight) * vert->pos + (taubin_weight / total_weight) * center;
        }

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
        {
            mesh->get_vertex(i)->pos = new_positions[i];
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////
template <typename Real>
void mesh_taubin_smoothing(MeshLib::Mesh3D<Real> *mesh, int iterations, bool feature_aware, Real sharp_angle_in_degree)
{
    if (!mesh || iterations <= 0)
        return;

    mesh->reset_edges_tag(false);
    mesh->reset_vertices_tag(false);

    if (mesh->is_quad())
    {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
        {
            auto vert = mesh->get_vertex(i);
            auto is_on_boundary = mesh->is_on_boundary(vert);
            vert->tag = is_on_boundary ? vert->degree != 3 : vert->degree != 4;
        }
    }

    if (feature_aware)
    {
        constexpr auto pi = Real(3.14159265358979323846);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->get_num_of_edges(); i += 2)
        {
            auto edge = mesh->get_edge(i);
            auto face1 = edge->face;
            auto face2 = edge->pair->face;
            if (face1 && face2)
            {
                Real angle = std::acos(std::min(Real(1), std::max(-Real(1), face1->normal.Dot(face2->normal)))) * 180 / pi;
                if (angle >= sharp_angle_in_degree)
                {
                    edge->tag = true;
                    edge->pair->tag = true;
                }
            }
        }
    }

    for (int iter = 0; iter < iterations; iter++)
    {
        // mesh_winslow_smoothing<Real>(mesh, 1, Real(0.4507499669));
        // mesh_winslow_smoothing<Real>(mesh, 1, Real(-0.4720265626));
        mesh_laplacian_smoothing<Real>(mesh, 1, Real(0.4507499669));
        mesh_laplacian_smoothing<Real>(mesh, 1, Real(-0.4720265626));
    }
    mesh->update_normal();
}
/////////////////////////////////////////////////////////////////////////////////////////
// template instantiation
template void mesh_laplacian_smoothing<double>(MeshLib::Mesh3D<double> *mesh, int iterations, double taubin_weight);
template void mesh_winslow_smoothing<double>(MeshLib::Mesh3D<double> *mesh, int iterations, double taubin_weight);
template void mesh_taubin_smoothing<double>(MeshLib::Mesh3D<double> *mesh, int iterations, bool feature_aware, double sharp_angle_in_degree);
/////////////////////////////////////////////////////////////////////////////////////////