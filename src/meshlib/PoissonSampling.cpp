#include "PoissonSampling.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

////////////////////////////////////////////////
template <typename Real>
PoissonSampling<Real>::PoissonSampling(unsigned int seed)
{
    srand(seed);
}
////////////////////////////////////////////////
template <typename Real>
bool PoissonSampling<Real>::sampling(Real radius, MeshLib::Mesh3D<Real> *input_mesh, std::vector<TinyVector<Real, 3>> &samples, std::vector<ptrdiff_t> &sample_located_face_ids, std::vector<TinyVector<Real, 3>> *sample_bary_coords, int num_samples, int raw_sample_ratio)
{
    if (input_mesh == nullptr || num_samples <= 0)
        return false;

    const auto vertex_count = input_mesh->get_num_of_vertices();
    const auto face_count = input_mesh->get_num_of_faces();
    std::vector<TinyVector<Real, 3>> mesh_vertices(vertex_count);
    for (ptrdiff_t i = 0; i < vertex_count; ++i)
    {
        mesh_vertices[i] = input_mesh->get_vertex(i)->pos;
    }
    std::vector<ptrdiff_t> mesh_facets;
    if (input_mesh->is_tri())
        mesh_facets.reserve(face_count * 3);
    else
        mesh_facets.reserve(face_count * 6);

    for (ptrdiff_t i = 0; i < face_count; ++i)
    {
        auto face = input_mesh->get_face(i);
        auto edge = face->edge;
        if (face->valence == 3)
        {
            mesh_facets.emplace_back(edge->vert->id);
            mesh_facets.emplace_back(edge->next->vert->id);
            mesh_facets.emplace_back(edge->next->next->vert->id);
        }
        else if (face->valence == 4)
        {
            auto v0 = edge->vert, v1 = edge->next->vert, v2 = edge->next->next->vert, v3 = edge->next->next->next->vert;
            bool tag = quad_util_split_quad(v0->pos, v1->pos, v2->pos, v3->pos);
            if (tag)
            {
                mesh_facets.emplace_back(v0->id);
                mesh_facets.emplace_back(v1->id);
                mesh_facets.emplace_back(v2->id);
                mesh_facets.emplace_back(v0->id);
                mesh_facets.emplace_back(v2->id);
                mesh_facets.emplace_back(v3->id);
            }
            else
            {
                mesh_facets.emplace_back(v0->id);
                mesh_facets.emplace_back(v1->id);
                mesh_facets.emplace_back(v3->id);
                mesh_facets.emplace_back(v1->id);
                mesh_facets.emplace_back(v2->id);
                mesh_facets.emplace_back(v3->id);
            }
        }
        else
        {
            for (ptrdiff_t j = 0; j < face->valence - 2; ++j)
            {
                mesh_facets.emplace_back(face->edge->pair->vert->id);
                mesh_facets.emplace_back(edge->vert->id);
                mesh_facets.emplace_back(edge->next->vert->id);
                edge = edge->next;
            }
        }
    }
    return sampling(radius, mesh_vertices, mesh_facets, samples, sample_located_face_ids, sample_bary_coords, num_samples, raw_sample_ratio);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool PoissonSampling<Real>::sampling(Real radius, const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<ptrdiff_t> &triangles, std::vector<TinyVector<Real, 3>> &samples, std::vector<ptrdiff_t> &sample_located_face_ids, std::vector<TinyVector<Real, 3>> *sample_bary_coords, int num_samples, int raw_sample_ratio)
{
    if (vertices.empty() || triangles.empty() || num_samples <= 0)
        return false;

    std::vector<Sample_Point<Real>> raw_samples(num_samples);
    std::vector<TinyVector<Real, 3>> tri_normal;

    Real surface_area = create_raw_samples(raw_sample_ratio * num_samples, vertices, triangles, raw_samples, tri_normal);

    if (radius <= 0)
        radius = std::sqrt(surface_area / num_samples) * static_cast<Real>(0.75);

    poisson_disk_from_samples(radius, tri_normal, raw_samples, samples, sample_located_face_ids, sample_bary_coords);

    return true;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void PoissonSampling<Real>::area_of_triangle(const TinyVector<Real, 3> &a, const TinyVector<Real, 3> &b, const TinyVector<Real, 3> &c, Real &area, TinyVector<Real, 3> &trinormal)
{
    trinormal = (b - a).Cross(c - a);
    area = trinormal.Normalize() / 2;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real PoissonSampling<Real>::random_value(Real maximum)
{
    return (maximum * rand()) / RAND_MAX;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real PoissonSampling<Real>::approximate_geodesic_distance(const TinyVector<Real, 3> &p1, const TinyVector<Real, 3> &p2, const TinyVector<Real, 3> &n1, const TinyVector<Real, 3> &n2)
{
    auto v = p2 - p1;
    auto l = v.Normalize();
    auto c1 = std::min(static_cast<Real>(1), std::max(static_cast<Real>(-1), n1.Dot(v)));
    auto c2 = std::min(static_cast<Real>(1), std::max(static_cast<Real>(-1), n2.Dot(v)));
    auto result = l * l;
    if (std::fabs(c1 - c2) > 1e-6)
        result *= (std::asin(c1) - std::asin(c2)) / (c1 - c2);
    return result;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real PoissonSampling<Real>::create_raw_samples(int num_samples,
                                               const std::vector<TinyVector<Real, 3>> &vertices,
                                               const std::vector<ptrdiff_t> &triangles,
                                               std::vector<Sample_Point<Real>> &samples,
                                               std::vector<TinyVector<Real, 3>> &tri_normal)
{
    std::vector<Real> tri_area(triangles.size() / 3);
    tri_normal.resize(triangles.size() / 3);
    std::map<Real, size_t> area_sum_to_index;
    Real max_area_sum = 0;
    for (size_t i = 0; i < triangles.size(); i += 3)
    {
        area_of_triangle(vertices[triangles[i]], vertices[triangles[i + 1]], vertices[triangles[i + 2]],
                         tri_area[i / 3], tri_normal[i / 3]);
        area_sum_to_index.emplace(max_area_sum, i / 3);
        max_area_sum += tri_area[i / 3];
    }

    samples.resize(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        auto r = random_value(max_area_sum);
        auto it = area_sum_to_index.upper_bound(r);
        --it;
        auto tri_index = it->second;
        auto tri = tri_index * 3;
        const auto &v0 = vertices[triangles[tri]];
        const auto &v1 = vertices[triangles[tri + 1]];
        const auto &v2 = vertices[triangles[tri + 2]];

        Real u = random_value(1), v = random_value(1);
        auto sqrt_u = std::sqrt(u);
        auto pos = v0 * (1 - sqrt_u) + v1 * (sqrt_u * (1 - v)) + v2 * (v * sqrt_u);
        samples[i].tri_id = tri_index;
        samples[i].cell_id = -1;
        samples[i].pos = pos;
        samples[i].bary_coords[0] = 1 - sqrt_u;
        samples[i].bary_coords[1] = sqrt_u * (1 - v);
        samples[i].bary_coords[2] = v * sqrt_u;
    }
    return max_area_sum;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void PoissonSampling<Real>::poisson_disk_from_samples(const Real radius,
                                                      const std::vector<TinyVector<Real, 3>> &tri_normal,
                                                      std::vector<Sample_Point<Real>> &raw_samples,
                                                      std::vector<TinyVector<Real, 3>> &samples,
                                                      std::vector<ptrdiff_t> &sample_located_face_ids,
                                                      std::vector<TinyVector<Real, 3>> *sample_bary_coords)
{
    TinyVector<Real, 3> min_bound(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max()),
        max_bound(-std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max());
    for (const auto &sample : raw_samples)
    {
        min_bound[0] = std::min(min_bound[0], sample.pos[0]);
        min_bound[1] = std::min(min_bound[1], sample.pos[1]);
        min_bound[2] = std::min(min_bound[2], sample.pos[2]);
        max_bound[0] = std::max(max_bound[0], sample.pos[0]);
        max_bound[1] = std::max(max_bound[1], sample.pos[1]);
        max_bound[2] = std::max(max_bound[2], sample.pos[2]);
    }
    TinyVector<Real, 3> boxsize = max_bound - min_bound;

    Real radius_square = radius * radius;
    TinyVector<Real, 3> grid_size = boxsize / radius;
    int grid_size_int[3] = {static_cast<int>(std::ceil(grid_size[0])), static_cast<int>(std::ceil(grid_size[1])), static_cast<int>(std::ceil(grid_size[2]))};
    grid_size_int[0] = std::max(1, grid_size_int[0]);
    grid_size_int[1] = std::max(1, grid_size_int[1]);
    grid_size_int[2] = std::max(1, grid_size_int[2]);

    grid_size[0] = boxsize[0] / grid_size_int[0];
    grid_size[1] = boxsize[1] / grid_size_int[1];
    grid_size[2] = boxsize[2] / grid_size_int[2];

    for (auto &sample : raw_samples)
    {
        auto rp = sample.pos - min_bound;
        auto ix = static_cast<int>(std::floor(rp[0] / grid_size[0]));
        auto iy = static_cast<int>(std::floor(rp[1] / grid_size[1]));
        auto iz = static_cast<int>(std::floor(rp[2] / grid_size[2]));
        sample.cell_id = ix + grid_size_int[0] * (iy + grid_size_int[1] * iz);
    }

    std::sort(raw_samples.begin(), raw_samples.end(), [](const Sample_Point<Real> &a, const Sample_Point<Real> &b)
              { return a.cell_id < b.cell_id; });

    std::unordered_map<ptrdiff_t, hash_data<Real>> cells;
    cells.reserve(raw_samples.size());

    ptrdiff_t last_id = -1;
    typename std::unordered_map<ptrdiff_t, hash_data<Real>>::iterator last_id_it;
    for (size_t i = 0; i < raw_samples.size(); ++i)
    {
        const auto &sample = raw_samples[i];
        if (sample.cell_id == last_id)
        {
            ++last_id_it->second.sample_cnt;
            continue;
        }
        hash_data<Real> data;
        data.first_sample_idx = i;
        data.sample_cnt = 1;

        auto result = cells.emplace(sample.cell_id, data);
        last_id = sample.cell_id;
        last_id_it = result.first;
    }

    std::vector<ptrdiff_t> neighbor_cell_offsets;
    neighbor_cell_offsets.reserve(27);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                auto offset = x + grid_size_int[0] * (y + grid_size_int[1] * z);
                neighbor_cell_offsets.emplace_back(offset);
            }
        }
    }

    int max_trials = 5;
    for (int trial = 0; trial < max_trials; ++trial)
    {
        for (auto &[cell_id, data] : cells)
        {
            ptrdiff_t next_sample_idx = data.first_sample_idx + trial;
            if (trial >= data.sample_cnt)
                continue;

            const auto &candidate = raw_samples[next_sample_idx];

            bool conflict = false;
            for (ptrdiff_t neighbor_offset : neighbor_cell_offsets)
            {
                ptrdiff_t neighbor_cell_id = cell_id + neighbor_offset;
                const auto neighbor_it = cells.find(neighbor_cell_id);
                if (neighbor_it == cells.end())
                    continue;

                const hash_data<Real> &neighbor_data = neighbor_it->second;
                for (const auto &sample : neighbor_data.poisson_samples)
                {
                    if ((sample.pos - candidate.pos).SquaredLength() < radius_square)
                    {
                        conflict = true;
                        break;
                    }
                }

                if (conflict)
                    break;
            }

            if (conflict)
                continue;

            data.poisson_samples.emplace_back(candidate);
        }
    }
    samples.clear();
    sample_located_face_ids.clear();
    samples.reserve(raw_samples.size());
    sample_located_face_ids.reserve(raw_samples.size());
    if (sample_bary_coords)
    {
        sample_bary_coords->clear();
        sample_bary_coords->reserve(raw_samples.size());
    }
    for (const auto &it : cells)
    {
        for (const auto &sample : it.second.poisson_samples)
        {
            samples.emplace_back(sample.pos);
            sample_located_face_ids.emplace_back(sample.tri_id);
            if (sample_bary_coords)
                sample_bary_coords->emplace_back(sample.bary_coords);
        }
    }
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real PoissonSampling<Real>::quad_util_compute_angle(const Real dot_product_value)
{
    return std::acos(std::min(static_cast<Real>(1), std::max(static_cast<Real>(-1), dot_product_value))) * 180 / static_cast<Real>(M_PI);
}

template <typename Real>
Real PoissonSampling<Real>::quad_util_compute_angle(const TinyVector<Real, 3> &unit_dir0, const TinyVector<Real, 3> &unit_dir1)
{
    return quad_util_compute_angle(unit_dir0.Dot(unit_dir1));
}

template <typename Real>
Real PoissonSampling<Real>::quad_util_compute_angle(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2)
{
    TinyVector<Real, 3> e0 = v0 - v1;
    TinyVector<Real, 3> e1 = v2 - v1;
    e0.Normalize();
    e1.Normalize();
    return quad_util_compute_angle(e0.Dot(e1));
};

template <typename Real>
bool PoissonSampling<Real>::quad_util_split_quad(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3)
{
    auto e01 = v0 - v1;
    auto e12 = v1 - v2;
    auto e23 = v2 - v3;
    auto e30 = v3 - v0;

    e01.Normalize(), e12.Normalize(), e23.Normalize(), e30.Normalize();
    auto sum02 = quad_util_compute_angle(e01, -e30) + quad_util_compute_angle(e12, -e23);
    auto sum13 = quad_util_compute_angle(e12, -e01) + quad_util_compute_angle(e23, -e30);

    return sum02 > sum13;
    // auto e01 = v1 - v0;
    // auto e02 = v2 - v0;
    // auto e03 = v3 - v0;
    // auto e12 = v2 - v1;
    // auto e13 = v3 - v1;
    // auto e23 = v3 - v2;
    // e01.Normalize(), e02.Normalize(), e03.Normalize(), e12.Normalize(), e13.Normalize(), e23.Normalize();
    // auto sum012 = quad_util_compute_angle(e01, e02) + quad_util_compute_angle(-e01, e12) + quad_util_compute_angle(-e02, -e12);
    // auto sum023 = quad_util_compute_angle(e02, e03) + quad_util_compute_angle(-e02, e23) + quad_util_compute_angle(-e03, -e23);
    // auto sum013 = quad_util_compute_angle(e01, e03) + quad_util_compute_angle(-e01, e13) + quad_util_compute_angle(-e03, -e13);
    // auto sum123 = quad_util_compute_angle(e12, e13) + quad_util_compute_angle(-e12, e23) + quad_util_compute_angle(-e13, -e23);
    // return sum012 + sum023 < sum013 + sum123;
};
//////////////////////////////////////////////////////////////////////////
template class PoissonSampling<double>;