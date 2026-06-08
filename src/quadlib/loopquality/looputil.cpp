
#include "looputil.h"
#include <numeric>
#include "eigen_3x3.h"
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real compute_angle(const Real dot_product_value)
{
    return std::acos(std::min((Real)1, std::max(-(Real)1, dot_product_value))) * 180 / (Real)M_PI;
}

template <typename Real>
Real compute_angle(const TinyVector<Real, 3> &unit_dir0, const TinyVector<Real, 3> &unit_dir1)
{
    return compute_angle(unit_dir0.Dot(unit_dir1));
}

template <typename Real>
Real compute_angle(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2)
{
    TinyVector<Real, 3> e0 = v0 - v1;
    TinyVector<Real, 3> e1 = v2 - v1;
    e0.Normalize();
    e1.Normalize();
    return compute_angle(e0.Dot(e1));
};

//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real compute_dihedral_angle(MeshLib::HE_edge<Real> *edge)
{
    auto face0 = edge->face;
    auto face1 = edge->pair->face;
    if (face0 == 0 || face1 == 0)
        return 0;
    auto b0 = edge->vert->pos - edge->pair->vert->pos;
    b0.Normalize();
    auto b1 = face0->GetCentroid() - edge->pair->vert->pos;
    auto b2 = face1->GetCentroid() - edge->pair->vert->pos;
    b1 -= b1.Dot(b0) * b0, b1.Normalize();
    b2 -= b2.Dot(b0) * b0, b2.Normalize();
    return compute_angle(b1, b2);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real compute_dihedral_angle_diag02(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3)
{
    auto b0 = v2 - v0;
    b0.Normalize();
    auto face023_centroid = (v0 + v2 + v3) / (Real)3.0;
    auto face012_centroid = (v0 + v1 + v2) / (Real)3.0;
    auto b1 = face023_centroid - v0;
    auto b2 = face012_centroid - v0;
    b1 -= b1.Dot(b0) * b0, b1.Normalize();
    b2 -= b2.Dot(b0) * b0, b2.Normalize();
    return compute_angle(b1, b2);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real quad_dihedral_angle_via_split_quad(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3)
{
    return std::max(compute_dihedral_angle_diag02(v0, v1, v2, v3), compute_dihedral_angle_diag02(v1, v2, v3, v0));
}

template <typename Real>
bool split_quad(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3)
{
    auto e01 = v0 - v1;
    auto e12 = v1 - v2;
    auto e23 = v2 - v3;
    auto e30 = v3 - v0;

    e01.Normalize(), e12.Normalize(), e23.Normalize(), e30.Normalize();
    auto sum02 = compute_angle(e01, -e30) + compute_angle(e12, -e23);
    auto sum13 = compute_angle(e12, -e01) + compute_angle(e23, -e30);

    return sum02 > sum13;

    // auto e01 = v1 - v0;
    // auto e02 = v2 - v0;
    // auto e03 = v3 - v0;
    // auto e12 = v2 - v1;
    // auto e13 = v3 - v1;
    // auto e23 = v3 - v2;
    // e01.Normalize(), e02.Normalize(), e03.Normalize(), e12.Normalize(), e13.Normalize(), e23.Normalize();
    // auto sum012 = compute_angle(e01, e02) + compute_angle(-e01, e12) + compute_angle(-e02, -e12);
    // auto sum023 = compute_angle(e02, e03) + compute_angle(-e02, e23) + compute_angle(-e03, -e23);
    // auto sum013 = compute_angle(e01, e03) + compute_angle(-e01, e13) + compute_angle(-e03, -e13);
    // auto sum123 = compute_angle(e12, e13) + compute_angle(-e12, e23) + compute_angle(-e13, -e23);
    // return sum012 + sum023 < sum013 + sum123;
}

template <typename Real>
TinyVector<Real, 3> rotate_along_axes(const TinyVector<Real, 3> &V, const TinyVector<Real, 3> &unit_axes, const Real angle)
{
    const Real cos_angle = std::cos(angle);
    const Real sin_angle = std::sin(angle);
    // return cos_angle * V + (1 - cos_angle) * V.Dot(unit_axes) * unit_axes + sin_angle * unit_axes.Cross(V);
    auto WX = unit_axes.Cross(V);
    return V + sin_angle * WX + (1 - cos_angle) * unit_axes.Cross(WX);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real quad_regularity(const TinyVector<Real, 3> &v0, const TinyVector<Real, 3> &v1, const TinyVector<Real, 3> &v2, const TinyVector<Real, 3> &v3)
{
    Real angle[4];
    angle[0] = compute_angle(v3, v0, v1);
    angle[1] = compute_angle(v0, v1, v2);
    angle[2] = compute_angle(v1, v2, v3);
    angle[3] = compute_angle(v2, v3, v0);
    Real sum = 0;
    for (int j = 0; j < 4; j++)
        sum += fabs(angle[j] - 90);
    return sum;
};
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void compute_statistics(const std::vector<Real> &data, Real &avg_v, Real &max_v, Real &var_v)
{
    if (data.empty())
    {
        avg_v = max_v = var_v = 0;
        return;
    }
    avg_v = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    max_v = *std::max_element(data.begin(), data.end());
    var_v = 0;
    for (ptrdiff_t i = 0; i < (ptrdiff_t)data.size(); i++)
    {
        var_v += (data[i] - avg_v) * (data[i] - avg_v);
    }
    var_v = std::sqrt(var_v / data.size());
};
//////////////////////////////////////////////////////////////////////////

template <typename Real>
void FittingPlane(const std::vector<TinyVector<Real, 3>> &points, const bool closed, TinyVector<Real, 3> &line_direction, TinyVector<Real, 3> &line_origin)
{
    if (points.size() < 3)
        return;
    TinyVector<Real, 3> cen(0, 0, 0);
    for (size_t i = 0; i < points.size(); i++)
        cen += points[i];
    line_origin = cen / (Real)points.size();

    if (points.size() == 3)
    {
        line_direction = (points[0] - points[1]).UnitCross(points[1] - points[2]);
        return;
    }
    TinyVector<Real, 3> N(0, 0, 0);

    for (size_t i = 0; i < (closed ? points.size() - 1 : points.size()); i++)
        N += points[i].Cross(points[(i + 1) % points.size()]);
    N.Normalize();
    line_direction = N;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void get_meshinfo(MeshLib::Mesh3D<Real> *mesh,
                  MySortedTuple<ptrdiff_t, 11, false> &info_tuple,
                  TinyVector<Real, 3> &size, bool apply_transformation)
{
    if (mesh == 0)
        return;

    auto nv = mesh->get_num_of_vertices();
    auto nf = mesh->get_num_of_faces();
    auto ne = mesh->get_num_of_edges();
    auto nb = mesh->get_num_of_boundaries();
    ptrdiff_t num_boundary_verts = 0;
    ptrdiff_t num_singular_verts = 0;
    ptrdiff_t num_interior_singular_verts = 0;
    ptrdiff_t num_boundary_singular_verts = 0;
    ptrdiff_t num_interior_degree_3_verts = 0;
    ptrdiff_t num_interior_degree_5_verts = 0;
    ptrdiff_t num_boundary_degree_2_verts = 0;
    for (auto i = 0; i < nv; i++)
    {
        auto vert = mesh->get_vertex(i);
        bool boundary = mesh->is_on_boundary(vert);
        if (boundary)
        {
            num_boundary_verts++;
            if (vert->degree == 2)
            {
                num_boundary_degree_2_verts++;
                num_boundary_singular_verts++;
            }
            else if (vert->degree > 3)
            {
                num_boundary_singular_verts++;
            }
        }
        else
        {
            if (vert->degree != 4)
            {
                num_interior_singular_verts++;
            }
            else if (vert->degree == 3)
            {
                num_interior_degree_3_verts++;
            }
            else if (vert->degree == 5)
            {
                num_interior_degree_5_verts++;
            }
        }
    }
    num_singular_verts = num_interior_singular_verts + num_boundary_singular_verts;

    info_tuple[0] = nv;
    info_tuple[1] = nf;
    info_tuple[2] = ne;
    info_tuple[3] = nb;
    info_tuple[4] = num_boundary_verts;
    info_tuple[5] = num_singular_verts;
    info_tuple[6] = num_interior_singular_verts;
    info_tuple[7] = num_boundary_singular_verts;
    info_tuple[8] = num_interior_degree_3_verts;
    info_tuple[9] = num_interior_degree_5_verts;
    info_tuple[10] = num_boundary_degree_2_verts;

    //////////////////////////////////////////////////////////

    Real x = 0, y = 0, z = 0;
#pragma omp parallel for reduction(+ : x, y, z)
    for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
    {
        auto vert = mesh->get_vertex(i);
        x += vert->pos[0], y += vert->pos[1], z += vert->pos[2];
    }
    TinyVector<Real, 3> normalization_center(x, y, z);
    normalization_center /= (Real)mesh->get_num_of_vertices();

    Real a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
#pragma omp parallel for reduction(+ : a00, a01, a02, a11, a12, a22)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)mesh->get_num_of_vertices(); i++)
    {
        auto p = mesh->get_vertex(i)->pos - normalization_center;
        a00 += p[0] * p[0];
        a01 += p[0] * p[1];
        a02 += p[0] * p[2];
        a11 += p[1] * p[1];
        a12 += p[1] * p[2];
        a22 += p[2] * p[2];
    }
    Real max_element = std::max({fabs(a00), fabs(a11), fabs(a22), fabs(a11), fabs(a12), fabs(a01)});
    if (max_element > 0)
    {
        a00 /= max_element;
        a01 /= max_element;
        a02 /= max_element;
        a11 /= max_element;
        a12 /= max_element;
        a22 /= max_element;
    }
    else
    {
        a00 = 1, a01 = 0, a02 = 0, a11 = 1, a12 = 0, a22 = 1; // if all the points are the same
    }
    Real pca_matrix[3][3];
    pca_matrix[0][0] = a00, pca_matrix[0][1] = pca_matrix[1][0] = a01, pca_matrix[0][2] = pca_matrix[2][0] = a02;
    pca_matrix[1][1] = a11, pca_matrix[1][2] = pca_matrix[2][1] = a12;
    pca_matrix[2][2] = a22;

    Real V[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Real D[3] = {0, 0, 0};
    Eigen3x3<Real> eigen(pca_matrix, V, D);

    TinyVector<Real, 3> T0(V[0][0], V[1][0], V[2][0]);
    TinyVector<Real, 3> T1(V[0][1], V[1][1], V[2][1]);
    TinyVector<Real, 3> T2(V[0][2], V[1][2], V[2][2]);
    if (D[0] > D[1])
        std::swap(T0, T1);
    if (D[1] > D[2])
        std::swap(T1, T2);
    if (D[0] > D[1])
        std::swap(T0, T1);

    if (T2.Dot(T0.Cross(T1)) < 0)
    {
        T2 *= (Real)-1;
    }

    TinyVector<Real, 3> FaceCenter(0, 0, 0);
    Real total_area = 0;
    for (ptrdiff_t i = 0; i < mesh->get_num_of_faces(); i++)
    {
        auto face = mesh->get_face(i);
        auto area = face->GetArea();
        FaceCenter += area * face->GetCentroid();
        total_area += area;
    }
    if (total_area != 0)
        FaceCenter /= total_area;

    auto dir = FaceCenter - normalization_center;
    if (dir.Dot(T0) < 0)
    {
        T0 *= (Real)-1;
        if (T1.Dot(dir) < 0)
            T1 *= (Real)-1;
        else
            T2 *= (Real)-1;
    }
    else
    {
        if (T1.Dot(dir) < 0)
        {
            T1 *= (Real)-1;
            T2 *= (Real)-1;
        }
    }

    auto left_lower_corner = TinyVector<Real, 3>(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max());
    auto right_upper_corner = TinyVector<Real, 3>(std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
    {
        auto vert = mesh->get_vertex(i);
        auto pos = vert->pos - normalization_center;
        pos = TinyVector<Real, 3>(T0 * pos, T1 * pos, T2 * pos);
        if (apply_transformation)
            vert->pos = pos;
#pragma omp critical
        {
            left_lower_corner[0] = std::min(left_lower_corner[0], pos[0]);
            left_lower_corner[1] = std::min(left_lower_corner[1], pos[1]);
            left_lower_corner[2] = std::min(left_lower_corner[2], pos[2]);
            right_upper_corner[0] = std::max(right_upper_corner[0], pos[0]);
            right_upper_corner[1] = std::max(right_upper_corner[1], pos[1]);
            right_upper_corner[2] = std::max(right_upper_corner[2], pos[2]);
        }
    }
    Real scale = 1.0 / (std::max(std::max(right_upper_corner[0] - left_lower_corner[0], right_upper_corner[1] - left_lower_corner[1]), right_upper_corner[2] - left_lower_corner[2]) + (Real)1.0e-10);
    left_lower_corner *= scale;
    right_upper_corner *= scale;
    size = right_upper_corner - left_lower_corner;

    if (apply_transformation)
    {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
        {
            mesh->get_vertex(i)->pos *= scale;
        }
        mesh->xmax = right_upper_corner[0];
        mesh->xmin = left_lower_corner[0];
        mesh->ymax = right_upper_corner[1];
        mesh->ymin = left_lower_corner[1];
        mesh->zmax = right_upper_corner[2];
        mesh->zmin = left_lower_corner[2];
        mesh->update_normal();
    }
}
//////////////////////////////////////////////////
template <typename Real>
bool skip_quad_mesh(MeshLib::Mesh3D<Real> *quad_mesh, size_t &num_interior_singularities, size_t &num_boundary_singularities, Real &dangle_diff, int &max_boundary_degree, int &max_interior_degree, int &max_num_of_singular_vertices_in_face)
{
    if (quad_mesh == 0)
        return true;

    max_boundary_degree = max_interior_degree = max_num_of_singular_vertices_in_face = 0;

    num_interior_singularities = num_boundary_singularities = 0;
    dangle_diff = 0;
    bool has_interior_singularies = false;
    bool has_boundary_singularies = false;
    unsigned int boundary_max_degree = 0;

    std::vector<bool> boundary_vertex_tags(quad_mesh->get_num_of_vertices()), singularity_tags(quad_mesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = quad_mesh->get_vertex(i);
        boundary_vertex_tags[i] = quad_mesh->is_on_boundary(hv);
        singularity_tags[i] = boundary_vertex_tags[i] ? hv->degree != 3 : hv->degree != 4;

        if (boundary_vertex_tags[i])
        {
            if (singularity_tags[i])
                num_boundary_singularities++;
            has_boundary_singularies |= singularity_tags[i];
            max_boundary_degree = std::max(max_boundary_degree, (int)hv->degree);
        }
        else
        {
            if (singularity_tags[i])
                num_interior_singularities++;
            has_interior_singularies |= singularity_tags[i];
            max_interior_degree = std::max(max_interior_degree, (int)hv->degree);
        }
    }
    // if (boundary_max_degree > 3)
    // {
    //     // std::cout << "Boundary max degree: " << boundary_max_degree << std::endl;
    //     return true;
    // }

    // if (!has_interior_singularies)
    // {
    //     // std::cout << "No interior singularities found." << std::endl;
    //     return true;
    // }

    // if (!has_boundary_singularies && !has_interior_singularies)
    //     return true;
    dangle_diff = 0;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto hf = quad_mesh->get_face(i);
        auto he = hf->edge;
        int sum = 0;
        do
        {
            if (singularity_tags[he->vert->id])
            {
                sum++;
            }
            he = he->next;
        } while (he != hf->edge);

        max_num_of_singular_vertices_in_face = std::max(max_num_of_singular_vertices_in_face, sum);
        // if (sum > 2)
        // {
        //     // std::cout << "Face " << i << " has " << sum << " singular vertices." << std::endl;
        //     return true;
        // }
        auto dangle = quad_dihedral_angle_via_split_quad(he->vert->pos, he->next->vert->pos, he->next->next->vert->pos, he->pair->vert->pos);
        dangle_diff = std::max(dangle_diff, 180 - dangle);
    }

    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_edges(); i++)
    {
        auto he = quad_mesh->get_edge(i);
        if (he < he->pair && quad_mesh->is_on_boundary(he) == false &&
            boundary_vertex_tags[he->vert->id] && boundary_vertex_tags[he->pair->vert->id])
        {
            // std::cout << "inerior Edge connects to two boundary vertices." << std::endl;
            return true;
        }
    }

    return false;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool has_zero_length_edge(MeshLib::Mesh3D<Real> *mesh, const Real tol)
{
    for (ptrdiff_t i = 0; i < mesh->get_num_of_edges(); i++)
    {
        auto he = mesh->get_edge(i);
        if (he < he->pair && he->GetLength() <= tol)
            return true;
    }
    return false;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool quick_filter(MeshLib::Mesh3D<Real> *quad_mesh)
{   
    if (quad_mesh->get_num_of_boundaries() > 8)
        return true;
    size_t num_interior_singularities = 0, num_boundary_singularities = 0;
    double dangle_diff = 0;
    int max_boundary_degree = 0, max_interior_degree = 0, max_num_of_singular_vertices_in_face = 0;
    int has_interior_edge_connecting_with_boundary = 0;
    has_interior_edge_connecting_with_boundary = skip_quad_mesh(quad_mesh, num_interior_singularities, num_boundary_singularities, dangle_diff,
                                                                max_boundary_degree, max_interior_degree, max_num_of_singular_vertices_in_face);
    return (num_interior_singularities ==0) || (dangle_diff > 60);
}
//////////////////////////////////////////////////////////////////////////
template double compute_angle(const double dot_product_value);
template double compute_angle(const TinyVector<double, 3> &unit_dir0, const TinyVector<double, 3> &unit_dir1);
template double compute_angle(const TinyVector<double, 3> &v0, const TinyVector<double, 3> &v1, const TinyVector<double, 3> &v2);
template double compute_dihedral_angle(MeshLib::HE_edge<double> *edge);
template double compute_dihedral_angle_diag02(const TinyVector<double, 3> &v0, const TinyVector<double, 3> &v1, const TinyVector<double, 3> &v2, const TinyVector<double, 3> &v3);
template double quad_dihedral_angle_via_split_quad(const TinyVector<double, 3> &v0, const TinyVector<double, 3> &v1, const TinyVector<double, 3> &v2, const TinyVector<double, 3> &v3);
template bool split_quad(const TinyVector<double, 3> &v0, const TinyVector<double, 3> &v1, const TinyVector<double, 3> &v2, const TinyVector<double, 3> &v3);
template TinyVector<double, 3> rotate_along_axes(const TinyVector<double, 3> &V, const TinyVector<double, 3> &unit_axes, const double angle);
template double quad_regularity(const TinyVector<double, 3> &v0, const TinyVector<double, 3> &v1, const TinyVector<double, 3> &v2, const TinyVector<double, 3> &v3);
template void compute_statistics(const std::vector<double> &data, double &avg_v, double &max_v, double &var_v);
template void FittingPlane(const std::vector<TinyVector<double, 3>> &points, const bool closed, TinyVector<double, 3> &line_direction, TinyVector<double, 3> &line_origin);
template void get_meshinfo(MeshLib::Mesh3D<double> *mesh, MySortedTuple<ptrdiff_t, 11, false> &info_tuple, TinyVector<double, 3> &size, bool apply_transformation);
template bool skip_quad_mesh(MeshLib::Mesh3D<double> *quad_mesh, size_t &num_interior_singularities, size_t &num_boundary_singularities, double &dangle_diff, int &max_boundary_degree, int &max_interior_degree, int &max_num_of_singular_vertices_in_face);
template bool has_zero_length_edge(MeshLib::Mesh3D<double> *mesh, const double tol);
template bool quick_filter(MeshLib::Mesh3D<double> *quad_mesh);