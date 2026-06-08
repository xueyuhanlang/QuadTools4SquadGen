#include "PointUtil.h"
// #include "ANN/ANN.h"
#include "PoissonSampling.h"
#include "eigen_3x3.h"
#include <algorithm>
#include "nanoflann.hpp"

template <typename Real>
ptrdiff_t MergeSamePoints(const std::vector<TinyVector<Real, 3>> &points,
                          std::vector<ptrdiff_t> &merge2uniqueID_map,
                          std::vector<ptrdiff_t> &back2overlapID_map,
                          double DIST_THRES, KNN_ENGINE knn_engine)
{
//     if (knn_engine == ANN)
//     {
//         ANNpointArray dataPts;
//         int n_pt = (int)points.size();
//         dataPts = annAllocPts(n_pt, 3);

//         double xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX;
//         double xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;
// #pragma omp parallel for
//         for (int k = 0; k < n_pt; k++)
//         {
//             dataPts[k][0] = (double)points[k][0];
//             dataPts[k][1] = (double)points[k][1];
//             dataPts[k][2] = (double)points[k][2];
// #pragma omp critical
//             {
//                 xmin = std::min(xmin, dataPts[k][0]), ymin = std::min(ymin, dataPts[k][1]), zmin = std::min(zmin, dataPts[k][2]);
//                 xmax = std::max(xmax, dataPts[k][0]), ymax = std::max(ymax, dataPts[k][1]), zmax = std::max(zmax, dataPts[k][2]);
//             }
//         }
//         double xlen = xmax - xmin, ylen = ymax - ymin, zlen = zmax - zmin;
//         double max_len = std::max(std::max(xlen, std::max(ylen, zlen)), 1.0e-10);
// #pragma omp parallel for
//         for (int k = 0; k < n_pt; k++)
//         {
//             dataPts[k][0] = (dataPts[k][0] - xmin) / max_len;
//             dataPts[k][1] = (dataPts[k][1] - ymin) / max_len;
//             dataPts[k][2] = (dataPts[k][2] - zmin) / max_len;
//         }

//         ANNkd_tree *data_kdTree = new ANNkd_tree(dataPts, n_pt, 3);

//         merge2uniqueID_map.assign(n_pt, -1);
//         back2overlapID_map.assign(n_pt, -1);

//         double queryPt[3];         // query point
//         std::vector<int> nnIdx;    // allocate near neigh indices
//         std::vector<double> dists; // allocate near neighbor dists
//         nnIdx.reserve(64), dists.reserve(64);

//         size_t unique_pt_counter = 0;

//         for (int k = 0; k < n_pt; k++)
//         {
//             if (merge2uniqueID_map[k] >= 0)
//             {
//                 continue;
//             } // skip already-found overlap points
//             int K = std::min(8, n_pt);

//             queryPt[0] = (double)dataPts[k][0];
//             queryPt[1] = (double)dataPts[k][1];
//             queryPt[2] = (double)dataPts[k][2];
//             bool done = true;
//             while (done)
//             {
//                 nnIdx.resize(K);
//                 dists.resize(K);
//                 data_kdTree->annkSearch( // search
//                     &queryPt[0],         // query point
//                     K,                   // number of near neighbors
//                     &nnIdx[0],           // nearest neighbors (returned)
//                     &dists[0],           // distance (returned)
//                     0.0);
//                 if (dists[K - 1] < DIST_THRES)
//                 {
//                     // if all the found K points are within distance threshold, there might be more overlap points.
//                     if (K == (int)points.size())
//                         return -1;
//                     K = std::min((int)points.size(), 2 * K);
//                     continue;
//                 }
//                 while (dists[K - 1] > DIST_THRES)
//                 {
//                     K--;
//                 }

//                 for (int i = 0; i < K; i++)
//                 {
//                     merge2uniqueID_map[nnIdx[i]] = unique_pt_counter;
//                 }
//                 break;
//             }
//             back2overlapID_map[unique_pt_counter] = k;
//             unique_pt_counter++;
//         }

//         delete data_kdTree;
//         annDeallocPts(dataPts);
//         annClose();

//         return unique_pt_counter;
//     }
    // else if (knn_engine == NANOFLANN)
    {

        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> my_kd_tree_t;
        PointCloud cloud(points);

        my_kd_tree_t m_kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        m_kdtree.buildIndex();

        ptrdiff_t n_pt = (ptrdiff_t)points.size();
        merge2uniqueID_map.assign(n_pt, -1);
        back2overlapID_map.assign(n_pt, -1);

        size_t unique_pt_counter = 0;

        for (auto k = 0; k < n_pt; k++)
        {
            if (merge2uniqueID_map[k] >= 0)
            {
                continue;
            } // skip already-found overlap points
            int K = std::min(8, (int)n_pt);
            std::vector<size_t> nearest_point_id;
            std::vector<double> distance;
            bool done = true;
            while (done)
            {
                nanoflann::KNNResultSet<double> resultSet(K);
                nearest_point_id.resize(K);
                distance.resize(K);
                resultSet.init(nearest_point_id.data(), distance.data());
                m_kdtree.findNeighbors(resultSet, &cloud[k][0], nanoflann::SearchParameters(10));
                if (distance[K - 1] < DIST_THRES)
                {
                    // if all the found K points are within distance threshold, there might be more overlap points.
                    if (K == (int)points.size())
                        return -1;
                    K = std::min((int)points.size(), 2 * K);
                    continue;
                }
                while (distance[K - 1] > DIST_THRES)
                {
                    K--;
                }

                for (int i = 0; i < K; i++)
                {
                    merge2uniqueID_map[nearest_point_id[i]] = unique_pt_counter;
                }
                break;
            }
            back2overlapID_map[unique_pt_counter] = k;
            unique_pt_counter++;
        }

        return unique_pt_counter;
    }
    // else
    // {
    //     throw std::runtime_error("Unsupported KNN engine.");
    //     return 0;
    // }
}

//////////////////////////////////////////////////////////////////////////
template <typename Real>
void scale_and_PCA(const std::vector<TinyVector<Real, 3>> &sample_points,
                   TinyVector<Real, 3> rotation[3],
                   TinyVector<Real, 3> inverse_rotation[3],
                   TinyVector<Real, 3> &normalization_center,
                   Real &normalization_scale)
{
    Real x = 0, y = 0, z = 0;
#pragma omp parallel for reduction(+ : x, y, z)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)sample_points.size(); i++)
    {
        x += sample_points[i][0], y += sample_points[i][1], z += sample_points[i][2];
    }
    normalization_center[0] = x / (Real)sample_points.size();
    normalization_center[1] = y / (Real)sample_points.size();
    normalization_center[2] = z / (Real)sample_points.size();

    Real a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
#pragma omp parallel for reduction(+ : a00, a01, a02, a11, a12, a22)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)sample_points.size(); i++)
    {
        auto p = sample_points[i] - normalization_center;
        a00 += p[0] * p[0];
        a01 += p[0] * p[1];
        a02 += p[0] * p[2];
        a11 += p[1] * p[1];
        a12 += p[1] * p[2];
        a22 += p[2] * p[2];
    }
    Real pca_matrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    pca_matrix[0][0] = a00, pca_matrix[0][1] = pca_matrix[1][0] = a01, pca_matrix[0][2] = pca_matrix[2][0] = a02;
    pca_matrix[1][1] = a11, pca_matrix[1][2] = pca_matrix[2][1] = a12;
    pca_matrix[2][2] = a22;

    Real V[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Real D[3] = {0, 0, 0};
    Eigen3x3<Real> eigen(pca_matrix, V, D);

    TinyVector<Real, 3> T0(V[0][0], V[1][0], V[2][0]);
    TinyVector<Real, 3> T1(V[0][1], V[1][1], V[2][1]);
    TinyVector<Real, 3> T2(V[0][2], V[1][2], V[2][2]);
    if (T2.Dot(T0.Cross(T1)) < 0)
    {
        T0 *= (Real)-1, T1 *= (Real)-1, T2 *= (Real)-1;
    }

    Real xmin = std::numeric_limits<Real>::max(), ymin = std::numeric_limits<Real>::max(), zmin = std::numeric_limits<Real>::max();
    Real xmax = std::numeric_limits<Real>::min(), ymax = std::numeric_limits<Real>::min(), zmax = std::numeric_limits<Real>::min();
    normalization_scale = 0;
    for (ptrdiff_t i = 0; i < (ptrdiff_t)sample_points.size(); i++)
    {
        auto p = sample_points[i] - normalization_center;
        p[0] = T0 * p, p[1] = T1 * p, p[2] = T2 * p;
        normalization_scale = std::max(normalization_scale, p.Length());
        // xmin = std::min(xmin, p[0]), ymin = std::min(ymin, p[1]), zmin = std::min(zmin, p[2]);
        // xmax = std::max(xmax, p[0]), ymax = std::max(ymax, p[1]), zmax = std::max(zmax, p[2]);
    }
    // normalization_scale = std::max(std::max(xmax - xmin, ymax - ymin), zmax - zmin);
    inverse_rotation[0][0] = T0[0], inverse_rotation[0][1] = T1[0], inverse_rotation[0][2] = T2[0];
    inverse_rotation[1][0] = T0[1], inverse_rotation[1][1] = T1[1], inverse_rotation[1][2] = T2[1];
    inverse_rotation[2][0] = T0[2], inverse_rotation[2][1] = T1[2], inverse_rotation[2][2] = T2[2];
    rotation[0] = T0, rotation[1] = T1, rotation[2] = T2;
}

//////////////////////////////////////////////////////////////////////////
template <typename Real>
void scale_and_PCA(MeshLib::Mesh3D<Real> *mesh,
                   TinyVector<Real, 3> inverse_rotation[3],
                   TinyVector<Real, 3> &normalization_center,
                   Real &normalization_scale, const int num_samples)
{
    if (mesh == 0)
        return;

    PoissonSampling<Real> poisson_sampling;
    std::vector<TinyVector<Real, 3>> sample_points;
    std::vector<ptrdiff_t> sample_point_face_ids;
    poisson_sampling.sampling((Real)-1, mesh, sample_points, sample_point_face_ids, 0, num_samples);

    TinyVector<Real, 3> rotation[3];
    scale_and_PCA(sample_points, rotation, inverse_rotation, normalization_center, normalization_scale);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
    {
        auto vert = mesh->get_vertex(i);
        vert->pos -= normalization_center;
        vert->pos = TinyVector<Real, 3>(rotation[0] * vert->pos, rotation[1] * vert->pos, rotation[2] * vert->pos) / normalization_scale;
        vert->normal = TinyVector<Real, 3>(rotation[0] * vert->normal, rotation[1] * vert->normal, rotation[2] * vert->normal);
    }
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < mesh->get_num_of_faces(); i++)
    {
        auto face = mesh->get_face(i);
        face->normal = TinyVector<Real, 3>(rotation[0] * face->normal, rotation[1] * face->normal, rotation[2] * face->normal);
    }
    mesh->compute_boundingbox();
}
//////////////////////////////////////////////////////////////////////////

template ptrdiff_t MergeSamePoints<double>(const std::vector<TinyVector<double, 3>> &points,
                                           std::vector<ptrdiff_t> &merge2uniqueID_map,
                                           std::vector<ptrdiff_t> &back2overlapID_map,
                                           double DIST_THRES, KNN_ENGINE knn_engine);

template void scale_and_PCA<double>(const std::vector<TinyVector<double, 3>> &sample_points,
                                    TinyVector<double, 3> rotation[3], TinyVector<double, 3> inverse_rotation[3],
                                    TinyVector<double, 3> &normalization_center, double &normalization_scale);

template void scale_and_PCA<double>(MeshLib::Mesh3D<double> *mesh, TinyVector<double, 3> inverse_rotation[3],
                                    TinyVector<double, 3> &normalization_center, double &normalization_scale, const int num_samples);