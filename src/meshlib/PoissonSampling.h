#pragma once

#include "Mesh3D.h"

// adopted from https://github.com/zewt/maya-implicit-skinning/blob/master/src/meshes/vcg_lib/utils_sampling.cpp

template <typename Real>
struct Sample_Point
{
    ptrdiff_t tri_id;
    ptrdiff_t cell_id;
    TinyVector<Real, 3> pos;
    TinyVector<Real, 3> bary_coords;
};

template <typename Real>
struct hash_data
{
    // Resulting output sample points for this cell:
    std::vector<Sample_Point<Real>> poisson_samples;

    // Index into raw_samples:
    ptrdiff_t first_sample_idx;
    ptrdiff_t sample_cnt;
};

template <typename Real>
class PoissonSampling
{
public:
    PoissonSampling(unsigned int seed = 0);
    bool sampling(Real radius, MeshLib::Mesh3D<Real> *input_mesh, std::vector<TinyVector<Real, 3>> &samples, std::vector<ptrdiff_t> &sample_located_face_ids, std::vector<TinyVector<Real, 3>> *sample_bary_coords = 0, int num_samples = 1000, int raw_sample_ratio = 10);
    bool sampling(Real radius, const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<ptrdiff_t> &triangles, std::vector<TinyVector<Real, 3>> &samples, std::vector<ptrdiff_t> &sample_located_face_ids, std::vector<TinyVector<Real, 3>> *sample_bary_coords = 0, int num_samples = 1000, int raw_sample_ratio = 10);

protected:
    //////////////////////////////////////////////////////////////////////////
    void area_of_triangle(const TinyVector<Real, 3> &a, const TinyVector<Real, 3> &b, const TinyVector<Real, 3> &c, Real &area, TinyVector<Real, 3> &trinormal);
    Real random_value(Real maximum);
    Real approximate_geodesic_distance(const TinyVector<Real, 3> &p1, const TinyVector<Real, 3> &p2, const TinyVector<Real, 3> &n1, const TinyVector<Real, 3> &n2);
    Real create_raw_samples(int num_samples,
                            const std::vector<TinyVector<Real, 3>> &vertices,
                            const std::vector<ptrdiff_t> &triangles,
                            std::vector<Sample_Point<Real>> &samples,
                            std::vector<TinyVector<Real, 3>> &tri_normal);
    void poisson_disk_from_samples(const Real radius,
                                   const std::vector<TinyVector<Real, 3>> &tri_normal,
                                   std::vector<Sample_Point<Real>> &raw_samples,
                                   std::vector<TinyVector<Real, 3>> &samples,
                                   std::vector<ptrdiff_t> &sample_located_face_ids,
                                   std::vector<TinyVector<Real, 3>> *sample_bary_coords = 0);

    Real quad_util_compute_angle(const Real dot_product_value);

    Real quad_util_compute_angle(const TinyVector<Real, 3> &unit_dir0, const TinyVector<Real, 3> &unit_dir1);

    Real quad_util_compute_angle(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2);

    bool quad_util_split_quad(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3);
};
