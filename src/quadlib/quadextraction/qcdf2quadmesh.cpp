#include "qcdf2quadmesh.h"
#include "libnpy/npz.h"
#include <chrono>
// #include <unordered_set>
#include <set>
#include <omp.h>
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "MeshSubdivision.h"
#include "looputil.h"
#include "myutils.h"
#include "MyTuple.h"
#include "NonmanifoldProcess.h"
#include <random>
#include "mesh_repair.h"
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

using namespace std::chrono;
////////////////////////////////
template <typename Real>
QCDF2QuadMesh<Real>::QCDF2QuadMesh(MeshLib::Mesh3D<Real> *mesh, const std::string &featurefile, bool pattern_subdiv)
    : m_pmesh(mesh), pattern_subdivision(pattern_subdiv)
{
    auto start = high_resolution_clock::now();
    if (!load_feature(featurefile))
    {
        std::cerr << "Cannot load the feature file!" << std::endl;
        throw std::runtime_error("Cannot load the feature file!");
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
}
////////////////////////////////
template <typename Real>
QCDF2QuadMesh<Real>::~QCDF2QuadMesh()
{
}
/////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::set_verbose(bool status)
{
    m_verbose = status;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_debug_mode(int status)
{
    m_debug_mode = status;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_confusion_band(Real band)
{
    confuse_band = band;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_sharp_angle(const Real angle_in_dgreee)
{
    sharp_feature_angle = angle_in_dgreee;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_debug_dir(const std::string &dir)
{
    debug_dir = dir;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_ring_size(int size)
{
    ring_size = std::max(1, size);
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_normalize(bool status)
{
    normalize_input = status;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_improve_mode(bool status)
{
    improve_mode = status;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_subdiv_num(unsigned int num)
{
    subdiv_num = num;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_smooth_num(unsigned int num)
{
    smooth_num = num;
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_edge_collapse_ratio(const Real ratio)
{
    edge_collapse_ratio = std::min(std::max(ratio, (Real)0), (Real)1);
}
template <typename Real>
void QCDF2QuadMesh<Real>::set_edge_collapse_normal_threshold(const Real angle_in_degree)
{
    edge_collapse_normal_threshold = angle_in_degree;
}
////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::export_mesh(const FaceClusterType type, const std::string &filename)
{
    init();

    if (m_debug_mode)
    {
        // print all parameters
        std::cout << "-- QCDF2QuadMesh parameters --" << std::endl;
        std::cout << "Debug mode is on." << std::endl;
        std::cout << "Confusion band: " << confuse_band << std::endl;
        std::cout << "Sharp feature angle: " << sharp_feature_angle << std::endl;
        std::cout << "Ring size: " << ring_size << std::endl;
        std::cout << "Normalize input mesh: " << (normalize_input ? "true" : "false") << std::endl;
        std::cout << "Improve mode: " << (improve_mode ? "true" : "false") << std::endl;
        std::cout << "Subdivision number: " << subdiv_num << std::endl;
        std::cout << "Smoothing number: " << smooth_num << std::endl;
        std::cout << "Edge collapse ratio: " << edge_collapse_ratio << std::endl;
        std::cout << "Edge collapse normal threshold: " << edge_collapse_normal_threshold << std::endl;
        std::cout << "Pattern subdivision: " << (pattern_subdivision ? "true" : "false") << std::endl;
        std::cout << "------------------------------" << std::endl;
    }

    if (type == FaceClusterType::FC_UNDEFINED)
    {
        std::cerr << "Undefined face cluster type!" << std::endl;
        throw std::runtime_error("Undefined face cluster type!");
    }

    input_method_type = type;

    auto start = high_resolution_clock::now();
    quad_extraction(type);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (m_verbose)
    {
        std::cout << "quad extraction time: " << duration.count() << " ms" << std::endl;
        std::cout << quad_vertices.size() << " vertices and " << quad_faces.size() << " faces generated." << std::endl;
    }
    if (quad_faces.empty())
    {
        if (m_verbose)
            std::cerr << "No quad mesh is generated!" << std::endl;
    }
    else
    {
        auto ext = GetFileExtension(filename);
        if (ext == "ply")
        {
            SavePLYMesh_with_color(filename, quad_vertices, quad_faces);
        }
        else
            save_mesh(quad_vertices, quad_faces, filename);
    }
}
/////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::subdivide_trimesh(const int level)
{
    if (level <= 0)
        return;
    if (m_verbose)
        std::cout << "Performing subdivision with level " << level << " for increasing the resolution!" << std::endl;
    MeshLib::Mesh3D<Real> *mesh = m_pmesh;

    size_t div = (size_t)pow(4, level);
    ptrdiff_t nv = m_pmesh->get_num_of_vertices(), ne = m_pmesh->get_num_of_edges() / 2, nf = m_pmesh->get_num_of_faces();
    std::vector<Real> vertex_cdf_colors(nv, 0), vertex_dcdf_colors(nv, 0);
    std::vector<TinyVector<Real, 3>> vertex_cdf_offsets(nv), vertex_dcdf_offsets(nv);
    for (ptrdiff_t i = 0; i < nv; i++)
    {
        auto hv = mesh->get_vertex(i);
        auto edge = hv->edge;
        Real area_sum = 0;
        TinyVector<Real, 3> cdf_center, dcdf_center;
        do
        {
            if (edge->face)
            {
                vertex_cdf_colors[i] += face_areas[edge->face->id] * face_cdf_colors[edge->face->id];
                vertex_dcdf_colors[i] += face_areas[edge->face->id] * face_dcdf_colors[edge->face->id];
                area_sum += face_areas[edge->face->id];
                cdf_center += face_areas[edge->face->id] * (face_centroids[edge->face->id] + coffset[edge->face->id]);
                dcdf_center += face_areas[edge->face->id] * (face_centroids[edge->face->id] + doffset[edge->face->id]);
            }
            edge = edge->pair->next;
        } while (edge != hv->edge);
        if (area_sum > 0)
        {
            vertex_cdf_colors[i] /= area_sum;
            vertex_dcdf_colors[i] /= area_sum;
            vertex_cdf_offsets[i] = cdf_center / area_sum - hv->pos;
            vertex_dcdf_offsets[i] = dcdf_center / area_sum - hv->pos;
        }
    }

    for (int iter = 0; iter < level; iter++)
    {
        nv = nv + ne;
        ne = ne * 2 + nf * 3;
        nf = nf * 4;
    }

    face_cdf_colors.resize(nf), face_dcdf_colors.resize(nf);
    vertex_cdf_colors.resize(nv), vertex_dcdf_colors.resize(nv);
    vertex_cdf_offsets.resize(nv), vertex_dcdf_offsets.resize(nv);
    coffset.resize(nf), doffset.resize(nf);

    for (int iter = 0; iter < level; iter++)
    {
        MeshLib::MeshSubdivision<Real> sub(mesh);
        std::vector<std::vector<ptrdiff_t>> vert_color_comb;
        MeshLib::Mesh3D<Real> *submesh = sub.SplitTri(&vert_color_comb);

#pragma omp parallel for
        for (ptrdiff_t i = mesh->get_num_of_vertices(); i < (ptrdiff_t)vert_color_comb.size(); i++)
        {
            auto hv = submesh->get_vertex(i);
            auto comb = vert_color_comb[i];
            Real cdf_color = 0, dcdf_color = 0;
            TinyVector<Real, 3> cdf_center, dcdf_center;
            for (auto vid : comb)
            {
                cdf_color += vertex_cdf_colors[vid];
                dcdf_color += vertex_dcdf_colors[vid];
                cdf_center += vertex_cdf_offsets[vid] + mesh->get_vertex(vid)->pos;
                dcdf_center += vertex_dcdf_offsets[vid] + mesh->get_vertex(vid)->pos;
            }
            vertex_cdf_colors[i] = cdf_color / 2;
            vertex_dcdf_colors[i] = dcdf_color / 2;
            vertex_cdf_offsets[i] = cdf_center / 2 - hv->pos;
            vertex_dcdf_offsets[i] = dcdf_center / 2 - hv->pos;
        }
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < submesh->get_num_of_faces(); i++)
        {
            auto hf = submesh->get_face(i);
            auto he = hf->edge;
            face_cdf_colors[i] = (vertex_cdf_colors[he->vert->id] + vertex_cdf_colors[he->next->vert->id] + vertex_cdf_colors[he->next->next->vert->id]) / (Real)3.0;
            face_dcdf_colors[i] = (vertex_dcdf_colors[he->vert->id] + vertex_dcdf_colors[he->next->vert->id] + vertex_dcdf_colors[he->next->next->vert->id]) / (Real)3.0;
            coffset[i] = (vertex_cdf_offsets[he->vert->id] + vertex_cdf_offsets[he->next->vert->id] + vertex_cdf_offsets[he->next->next->vert->id]) / (Real)3.0;
            doffset[i] = (vertex_dcdf_offsets[he->vert->id] + vertex_dcdf_offsets[he->next->vert->id] + vertex_dcdf_offsets[he->next->next->vert->id]) / (Real)3.0;
        }
        auto max_cdf_color = *std::max_element(face_cdf_colors.begin(), face_cdf_colors.end());
        auto min_cdf_color = *std::min_element(face_cdf_colors.begin(), face_cdf_colors.end());
        auto max_dcdf_color = *std::max_element(face_dcdf_colors.begin(), face_dcdf_colors.end());
        auto min_dcdf_color = *std::min_element(face_dcdf_colors.begin(), face_dcdf_colors.end());
        auto cdf_range = max_cdf_color - min_cdf_color;
        auto dcdf_range = max_dcdf_color - min_dcdf_color;
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < submesh->get_num_of_faces(); i++)
        {
            face_cdf_colors[i] = (face_cdf_colors[i] - min_cdf_color) / cdf_range;
            face_dcdf_colors[i] = (face_dcdf_colors[i] - min_dcdf_color) / dcdf_range;
        }
        std::swap(mesh, submesh);
        if (m_pmesh != submesh)
            delete submesh;
    }
    m_pmesh = mesh;

    // update other attributes
    face_centroids.resize(nf);
    face_areas.resize(nf);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nf; i++)
    {
        auto hf = m_pmesh->get_face(i);
        face_centroids[i] = hf->GetCentroid();
        face_areas[i] = hf->GetArea();
        // coffset[i] = coffset[i / div], doffset[i] = doffset[i / div];
    }
}
////////////////////////////////
template <typename Real>
int QCDF2QuadMesh<Real>::identify_seed_faces(const FaceClusterType type, std::vector<FaceClusterType> &seed_face_tags)
{
    const std::vector<Real> &face_colors = (type == FaceClusterType::FC_DCDF) ? face_dcdf_colors : face_cdf_colors;
    seed_face_tags.assign(m_pmesh->get_num_of_faces(), FaceClusterType::FC_UNDEFINED);
    // to avoid the same color values, jitter the face colors a bit
    std::vector<Real> jittered_face_colors(m_pmesh->get_num_of_faces(), 0);
    // Use a fixed seed for reproducibility and a faster random number generator
    std::mt19937 rng(0);
    std::uniform_real_distribution<Real> dist(-0.5e-8, 0.5e-8);
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        Real jitter = dist(rng);
        // jittered_face_colors[i] = std::max(0.0, std::min(face_colors[i] + jitter, (Real)1.0));
        jittered_face_colors[i] = face_colors[i] + jitter;
    }

    // Real color_bound = (Real)0.1;
    int num_seeds = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : num_seeds)
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        // if (jittered_face_colors[i] < mid_color)
        // if (jittered_face_colors[i] < 0.1)
        // continue;
        // if (jittered_face_colors[i] < color_bound)
        //     continue;
        std::unordered_set<ptrdiff_t> visited;
        std::vector<ptrdiff_t> current_ring, next_ring;
        current_ring.push_back(i);
        visited.insert(i);
        bool stop_search = false;
        for (int ring = 0; ring < ring_size; ring++)
        {
            next_ring.resize(0);
            for (size_t j = 0; j < current_ring.size(); j++)
            {
                auto hf = m_pmesh->get_face(current_ring[j]);
                // if (jittered_face_colors[current_ring[j]] < color_bound)
                //     continue;
                auto he = hf->edge;
                do
                {
                    auto hv = he->vert;
                    auto hv_edge = hv->edge;
                    do
                    {
                        if (hv_edge->face)
                        {
                            ptrdiff_t neighbor_fid = hv_edge->face->id;
                            if (visited.find(neighbor_fid) == visited.end())
                            {
                                next_ring.push_back(neighbor_fid);
                                visited.insert(neighbor_fid);
                                if (jittered_face_colors[neighbor_fid] > jittered_face_colors[i])
                                {
                                    stop_search = true;
                                    break;
                                }
                            }
                        }
                        hv_edge = hv_edge->pair->next;
                    } while (hv_edge != hv->edge);
                    if (stop_search)
                        break;
                    he = he->next;
                } while (he != hf->edge);
                if (stop_search)
                    break;
            }
            if (stop_search)
                break;
            current_ring = next_ring;
        }
        if (!stop_search)
        {
            seed_face_tags[i] = type;
            num_seeds++;
        }
    }

    if (m_debug_mode)
    {
        std::vector<TinyVector<Real, 3>> seed_face_centers;
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
        {
            if (seed_face_tags[i] != FaceClusterType::FC_UNDEFINED)
            {
                seed_face_centers.push_back(face_centroids[i] + 0.001 * m_pmesh->get_face(i)->normal);
            }
        }
        std::string type_str = (type == FaceClusterType::FC_DCDF) ? "dcdf" : "cdf";
        SavePtsPLY(path_concatenate(debug_dir, "seed_face_centers_" + type_str + ".ply"), seed_face_centers);
    }

    return num_seeds;
}
template <typename Real>
void QCDF2QuadMesh<Real>::init()
{
    if (pattern_subdivision)
        color_pattern_subdivision();

    if (normalize_input)
    {
        // center and normalize the mesh for better visualization
        global_scale = std::max(std::max(m_pmesh->xmax - m_pmesh->xmin, m_pmesh->ymax - m_pmesh->ymin), m_pmesh->zmax - m_pmesh->zmin) / 10;
        TinyVector<Real, 3> center((m_pmesh->xmax + m_pmesh->xmin) / (Real)2,
                                   (m_pmesh->ymax + m_pmesh->ymin) / (Real)2,
                                   (m_pmesh->zmax + m_pmesh->zmin) / (Real)2);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
        {
            auto hv = m_pmesh->get_vertex(i);
            hv->pos = (hv->pos - center) / global_scale;
        }
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
        {
            coffset[i] /= global_scale;
            doffset[i] /= global_scale;
        }
        global_scale = 10;
    }
    else
    {
        global_scale = std::max(std::max(m_pmesh->xmax - m_pmesh->xmin, m_pmesh->ymax - m_pmesh->ymin), m_pmesh->zmax - m_pmesh->zmin);
    }

    auto nf = m_pmesh->get_num_of_faces();
    face_centroids.resize(nf);
    face_areas.resize(nf);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nf; i++)
    {
        auto hf = m_pmesh->get_face(i);
        face_centroids[i] = hf->GetCentroid();
        face_areas[i] = hf->GetArea();
    }

    trimesh_feature_edge_tags.assign(m_pmesh->get_num_of_edges(), 0);
    color_feature_edge_tags.assign(m_pmesh->get_num_of_edges(), 0);

#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge > edge->pair)
            continue;
        if (!m_pmesh->is_on_boundary(edge))
        {
            trimesh_feature_edge_tags[edge->id] = trimesh_feature_edge_tags[edge->pair->id] = compute_dihedral_angle(edge) <= sharp_feature_angle ? 1 : 0;
            if (trimesh_feature_edge_tags[edge->id] == 1)
            {
                if (face_cdf_colors[edge->face->id] <= feature_color_band && face_cdf_colors[edge->pair->face->id] <= feature_color_band)
                {
                    color_feature_edge_tags[edge->id] = color_feature_edge_tags[edge->pair->id] = 1;
                }
            }
        }
        else
            trimesh_feature_edge_tags[edge->id] = trimesh_feature_edge_tags[edge->pair->id] = 1;
    }

    extract_trimesh_featurelines(vertex_feature_tag, feature_edge_loops);

    // std::vector<bool> edge_tags(trimesh_feature_edge_tags.begin(), trimesh_feature_edge_tags.end());
    // SaveMarkedEdge_as_ply(m_pmesh, path_concatenate(debug_dir, "trimesh_feature_edges.ply"), edge_tags, global_scale * (Real)0.001);
    // std::vector<bool> edge_tags(color_feature_edge_tags.begin(), color_feature_edge_tags.end());
    // SaveMarkedEdge_as_ply(m_pmesh, path_concatenate(debug_dir, "color_feature_edges.ply"), edge_tags, global_scale * (Real)0.001);

    tri_mesh_vertex_cdfdcdf_types.assign(m_pmesh->get_num_of_vertices(), -1);
}
/////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::feature_edge_travel(MeshLib::HE_edge<Real> *start_edge,
                                              const std::vector<int> &vertex_feature_tag,
                                              std::vector<int> &feature_edge_tag,
                                              std::vector<MeshLib::HE_vert<Real> *> &loop_vertices)
{
    loop_vertices.resize(0);
    if (feature_edge_tag[start_edge->id] != 1)
        return;
    loop_vertices.push_back(start_edge->pair->vert);
    feature_edge_tag[start_edge->id] = feature_edge_tag[start_edge->pair->id] = 0; // set the edge as used
    auto end_vertex = start_edge->vert;
    bool stop = true;
    do
    {
        loop_vertices.push_back(end_vertex);
        if (vertex_feature_tag[end_vertex->id])
            break;
        auto vedge = end_vertex->edge;
        stop = true;
        do
        {
            if (feature_edge_tag[vedge->id])
            {
                end_vertex = vedge->vert;
                feature_edge_tag[vedge->id] = feature_edge_tag[vedge->pair->id] = 0; // set the edge as used
                stop = false;
                break;
            }
            vedge = vedge->pair->next;
        } while (vedge != end_vertex->edge);
    } while (!stop);
}
/////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::extract_trimesh_featurelines(
    std::vector<int> &vertex_feature_tag,
    std::vector<std::vector<MeshLib::HE_vert<Real> *>> &feature_edge_loops)
{
    std::vector<int> feature_edge_tag(trimesh_feature_edge_tags);

    vertex_feature_tag.assign(m_pmesh->get_num_of_vertices(), 0);
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto hv = m_pmesh->get_vertex(i);
        auto he = hv->edge;
        std::vector<MeshLib::HE_vert<Real> *> boundary_verts, interior_verts;
        do
        {
            if (!he->face || !he->pair->face)
                boundary_verts.push_back(he->vert);
            if (feature_edge_tag[he->id] && !m_pmesh->is_on_boundary(he))
                interior_verts.push_back(he->vert);
            he = he->pair->next;
        } while (he != hv->edge);

        if (boundary_verts.empty())
        {
            if (!interior_verts.empty())
            {
                if (interior_verts.size() != 2)
                    vertex_feature_tag[i] = 1;
                else
                {
                    vertex_feature_tag[i] = compute_angle(interior_verts.front()->pos, hv->pos, interior_verts.back()->pos) <= sharp_feature_angle ? 1 : 0;
                }
            }
        }
        else
        {
            if (!interior_verts.empty())
                vertex_feature_tag[i] = 1;
            else
            {
                vertex_feature_tag[i] = compute_angle(boundary_verts.front()->pos, hv->pos, boundary_verts.back()->pos) <= sharp_feature_angle ? 1 : 0;
            }
        }
    }

    // std::vector<TinyVector<Real, 3>> feature_points;
    // for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    // {
    //     if (vertex_feature_tag[i])
    //         feature_points.push_back(m_pmesh->get_vertex(i)->pos);
    // }
    // SavePtsPLY(path_concatenate(debug_dir, "trimesh_feature_vertices.ply"), feature_points);

    // std::cout << std::count(feature_edge_tag.begin(), feature_edge_tag.end(), 1) / 2<< " feature edges in trimesh" << std::endl;

    feature_edge_loops.resize(0);
    std::vector<MeshLib::HE_vert<Real> *> loop_vertices;

    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        if (!vertex_feature_tag[i])
            continue;
        auto vertex = m_pmesh->get_vertex(i);

        auto edge = vertex->edge;
        do
        {
            if (feature_edge_tag[edge->id])
            {
                feature_edge_travel(edge, vertex_feature_tag, feature_edge_tag, loop_vertices);
                if (!loop_vertices.empty())
                    feature_edge_loops.emplace_back(loop_vertices);
            }
            edge = edge->pair->next;
        } while (edge != vertex->edge);
    }
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); ++i)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge > edge->pair || feature_edge_tag[i] != 1)
            continue;

        feature_edge_travel(edge, vertex_feature_tag, feature_edge_tag, loop_vertices);
        feature_edge_loops.emplace_back(loop_vertices);
    }
    // std::cout << feature_edge_loops.size() << " feature edge loops in trimesh" << std::endl;
    // std::cout << std::count(vertex_feature_tag.begin(), vertex_feature_tag.end(), 1) << " feature vertices in trimesh" << std::endl;
}
/////////////////////////////////
template <typename Real>
bool QCDF2QuadMesh<Real>::load_feature(const std::string &featurefile)
{
    try
    {
        npy::inpzstream input(featurefile);
        if (!input.is_open())
        {
            std::cerr << "Cannot open the feature file!" << std::endl;
            return false;
        }

        auto cdf = input.read<float>("cdf.npy");
        auto dcdf = input.read<float>("dcdf.npy");

        bool has_vertex_info = false;
        if (cdf.shape()[0] == m_pmesh->get_num_of_vertices() + m_pmesh->get_num_of_faces())
            has_vertex_info = true;
        else if (cdf.shape()[0] != m_pmesh->get_num_of_faces())
        {
            std::cout << featurefile << " has " << cdf.shape()[0] << " entries, but the input mesh has " << m_pmesh->get_num_of_faces() << " faces and " << m_pmesh->get_num_of_vertices() << " vertices." << std::endl;
            throw std::runtime_error("The number of entries in the feature file does not match with the input mesh!");
        }
        int nfaces = (int)m_pmesh->get_num_of_faces();

        auto offsetd = input.read<float>("offset.npy");
        auto offsetc = input.read<float>("offsetc.npy");
        // auto offsetb = input.read<float>("offsetb.npy");
        // auto offseta = input.read<float>("offsetd.npy");

        face_cdf_colors.resize(nfaces), face_dcdf_colors.resize(nfaces);
        coffset.resize(nfaces), doffset.resize(nfaces);
        // boffset.resize(nfaces), aoffset.resize(nfaces);

#pragma omp parallel for
        for (int i = 0; i < nfaces; i++)
        {
            face_dcdf_colors[i] = (Real)dcdf(i, 0);
            face_cdf_colors[i] = (Real)cdf(i, 0);
            face_dcdf_colors[i] = std::clamp(face_dcdf_colors[i], (Real)0, (Real)1);
            face_cdf_colors[i] = std::clamp(face_cdf_colors[i], (Real)0, (Real)1);
            doffset[i][0] = (Real)offsetd(i, 0);
            doffset[i][1] = (Real)offsetd(i, 1);
            doffset[i][2] = (Real)offsetd(i, 2);
            coffset[i][0] = (Real)offsetc(i, 0);
            coffset[i][1] = (Real)offsetc(i, 1);
            coffset[i][2] = (Real)offsetc(i, 2);
            // boffset[i][0] = (Real)offsetb(i, 0);
            // boffset[i][1] = (Real)offsetb(i, 1);
            // boffset[i][2] = (Real)offsetb(i, 2);
            // aoffset[i][0] = (Real)offseta(i, 0);
            // aoffset[i][1] = (Real)offseta(i, 1);
            // aoffset[i][2] = (Real)offseta(i, 2);
        }

        //         Real max_dcdf_color = *std::max_element(face_dcdf_colors.begin(), face_dcdf_colors.end());
        //         Real min_dcdf_color = *std::min_element(face_dcdf_colors.begin(), face_dcdf_colors.end());
        //         Real max_cdf_color = *std::max_element(face_cdf_colors.begin(), face_cdf_colors.end());
        //         Real min_cdf_color = *std::min_element(face_cdf_colors.begin(), face_cdf_colors.end());
        //         Real dcdf_range = max_dcdf_color - min_dcdf_color;
        //         Real cdf_range = max_cdf_color - min_cdf_color;
        // #pragma omp parallel for
        //         for (ptrdiff_t i = 0; i < nfaces; i++)
        //         {
        //             face_dcdf_colors[i] = (face_dcdf_colors[i] - min_dcdf_color) / (dcdf_range + (Real)1e-8);
        //             face_cdf_colors[i] = (face_cdf_colors[i] - min_cdf_color) / (cdf_range + (Real)1e-8);
        //         }

        if (has_vertex_info)
        {
            int nverts = (int)m_pmesh->get_num_of_vertices();
            vertex_cdf_colors.resize(nverts);
            vertex_dcdf_colors.resize(nverts);
            v_coffset.resize(nverts);
            v_doffset.resize(nverts);
#pragma omp parallel for
            for (int i = 0; i < (int)nverts; i++)
            {
                vertex_dcdf_colors[i] = (Real)dcdf(nfaces + i, 0);
                vertex_cdf_colors[i] = (Real)cdf(nfaces + i, 0);
                vertex_dcdf_colors[i] = std::clamp(vertex_dcdf_colors[i], (Real)0, (Real)1);
                vertex_cdf_colors[i] = std::clamp(vertex_cdf_colors[i], (Real)0, (Real)1);
                v_doffset[i][0] = (Real)offsetd(nfaces + i, 0);
                v_doffset[i][1] = (Real)offsetd(nfaces + i, 1);
                v_doffset[i][2] = (Real)offsetd(nfaces + i, 2);
                v_coffset[i][0] = (Real)offsetc(nfaces + i, 0);
                v_coffset[i][1] = (Real)offsetc(nfaces + i, 1);
                v_coffset[i][2] = (Real)offsetc(nfaces + i, 2);
            }
            // clip the color values to [0,1]
            // std::clamp(vertex_dcdf_colors.begin(), vertex_dcdf_colors.end(), (Real)0, (Real)1);
            // std::clamp(vertex_cdf_colors.begin(), vertex_cdf_colors.end(), (Real)0, (Real)1);

            //             Real max_dcdf_color = *std::max_element(vertex_dcdf_colors.begin(), vertex_dcdf_colors.end());
            //             Real min_dcdf_color = *std::min_element(vertex_dcdf_colors.begin(), vertex_dcdf_colors.end());
            //             Real max_cdf_color = *std::max_element(vertex_cdf_colors.begin(), vertex_cdf_colors.end());
            //             Real min_cdf_color = *std::min_element(vertex_cdf_colors.begin(), vertex_cdf_colors.end());
            //             Real dcdf_range = max_dcdf_color - min_dcdf_color;
            //             Real cdf_range = max_cdf_color - min_cdf_color;
            // #pragma omp parallel for
            //             for (ptrdiff_t i = 0; i < nverts; i++)
            //             {
            //                 vertex_dcdf_colors[i] = (vertex_dcdf_colors[i] - min_dcdf_color) / (dcdf_range + (Real)1e-8);
            //                 vertex_cdf_colors[i] = (vertex_cdf_colors[i] - min_cdf_color) / (cdf_range + (Real)1e-8);
            //             }
        }

        auto invT = input.read<float>("invT.npy");

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                inverse_rotation[i][j] = (Real)invT(i, j);
            }
            normalization_center[i] = (Real)invT(i, 3);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}
////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::color_pattern_subdivision()
{
    int div = 1;
    ptrdiff_t nfaces = face_cdf_colors.size();
#pragma omp parallel for
    for (int i = 0; i < nfaces; i++)
    {
        Real u = 1 - face_cdf_colors[i], v = face_dcdf_colors[i];
        Real u_new = std::abs(u - (std::floor(u * div) + 0.5) / div);
        Real v_new = std::abs(v - (std::floor(v * div) + 0.5) / div);
        face_cdf_colors[i] = 1 - 2 * div * (u_new > v_new ? u_new : v_new);

        u_new = std::abs(u * div - (std::floor(u * div + 0.5)));
        v_new = std::abs(v * div - (std::floor(v * div + 0.5)));
        face_dcdf_colors[i] = 1 - 2 * (u_new > v_new ? u_new : v_new);
    }
    ptrdiff_t nverts = vertex_cdf_colors.size();
#pragma omp parallel for
    for (int i = 0; i < nverts; i++)
    {
        Real u = 1 - vertex_cdf_colors[i], v = vertex_dcdf_colors[i];
        Real u_new = std::abs(u - (std::floor(u * div) + 0.5) / div);
        Real v_new = std::abs(v - (std::floor(v * div) + 0.5) / div);
        vertex_cdf_colors[i] = 1 - 2 * div * (u_new > v_new ? u_new : v_new);

        u_new = std::abs(u * div - (std::floor(u * div + 0.5)));
        v_new = std::abs(v * div - (std::floor(v * div + 0.5)));
        vertex_dcdf_colors[i] = 1 - 2 * (u_new > v_new ? u_new : v_new);
    }
}
////////////////////////////////
template <typename Real>
Real QCDF2QuadMesh<Real>::distance_to_cluster(const ptrdiff_t fid, const ptrdiff_t cluster_centerface_id, const FaceClusterType type)
{
    auto v = face_centroids[fid] - face_centroids[cluster_centerface_id];
    if (pattern_subdivision)
    {
        if (type == FaceClusterType::FC_CDF)
            return (v + coffset[fid] - coffset[cluster_centerface_id]).Length() +
                   (v + doffset[fid] - doffset[cluster_centerface_id]).Length();
        else
            return (v + aoffset[fid] - aoffset[cluster_centerface_id]).Length() +
                   (v + boffset[fid] - boffset[cluster_centerface_id]).Length();
    }
    else
    {
        if (type == FaceClusterType::FC_DCDF)
            return (v + doffset[fid] - doffset[cluster_centerface_id]).Length();
        else // if (type == FaceClusterType::FC_CDF)
            return (v + coffset[fid] - coffset[cluster_centerface_id]).Length();
    }
}
////////////////////////////////
template <typename Real>
bool QCDF2QuadMesh<Real>::is_manifold_vertex(const ptrdiff_t vid)
{
    auto hv = m_pmesh->get_vertex(vid);
    std::vector<ptrdiff_t> incident_clusters;
    auto he = hv->edge;
    do
    {
        if (he->face)
        {
            auto cid = face_cluster_ids[he->face->id];
            if (cid > 0)
                incident_clusters.push_back(cid);
        }
        else
        {
            incident_clusters.push_back(-1);
        }
        he = he->pair->next;
    } while (he != hv->edge);

    if (incident_clusters.size() <= 1)
        return true;

    size_t split_location = 0;
    for (size_t i = 0; i < incident_clusters.size(); i++)
    {
        if (incident_clusters[i] != incident_clusters[(i - 1 + incident_clusters.size()) % incident_clusters.size()])
        {
            split_location = i;
            break;
        }
    }
    ptrdiff_t prev_cluster = incident_clusters[split_location];
    std::unordered_set<ptrdiff_t> unique_clusters;
    unique_clusters.insert(prev_cluster);
    for (size_t i = 1; i < incident_clusters.size(); i++)
    {
        size_t idx = (split_location + i) % incident_clusters.size();
        if (incident_clusters[idx] != prev_cluster)
        {
            if (unique_clusters.find(incident_clusters[idx]) != unique_clusters.end())
                return false;
            unique_clusters.insert(incident_clusters[idx]);
            prev_cluster = incident_clusters[idx];
        }
    }
    return true;
}
////////////////////////////////
template <typename Real>
bool QCDF2QuadMesh<Real>::has_nonmanifold_vertex(const ptrdiff_t face_id)
{
    auto face = m_pmesh->get_face(face_id);
    auto edge = face->edge;
    do
    {
        auto vid = edge->vert->id;
        if (!is_manifold_vertex(vid))
            return true;
        edge = edge->next;
    } while (edge != face->edge);
    return false;
}
////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::process_uncluster_faces(const FaceClusterType type, const int num_clusters)
{
    auto nf = m_pmesh->get_num_of_faces();

    const auto &face_offsets = (type == FaceClusterType::FC_DCDF) ? doffset : coffset;

    typedef std::tuple<MeshLib::HE_face<Real> *, Real, ptrdiff_t> UnClusterFace;
    class unclustered_face_compare
    {
    public:
        int operator()(const UnClusterFace &a, const UnClusterFace &b)
        {
            return std::get<1>(a) > std::get<1>(b);
        }
    };
    std::priority_queue<UnClusterFace, std::deque<UnClusterFace>, unclustered_face_compare> unclustered_faces;
    std::vector<Real> unclustered_face_diff_distance(nf, std::numeric_limits<Real>::max());

    std::vector<bool> unclustered_face_tags(nf, false);
#pragma omp parallel for
    for (auto i = 0; i < nf; i++)
    {
        if (face_cluster_ids[i] == 0)
        {
            unclustered_face_tags[i] = true;
        }
    }

    bool check_sharp_feature = (type == FaceClusterType::FC_CDF);

    for (auto i = 0; i < nf; i++)
    {
        if (unclustered_face_tags[i])
            continue;

        auto face = m_pmesh->get_face(i);
        auto edge = face->edge;
        auto cid = face_cluster_ids[face->id];
        do
        {
            if (edge->pair->face &&
                unclustered_face_tags[edge->pair->face->id] && (!check_sharp_feature || color_feature_edge_tags[edge->id] == false))
            {
                auto pair_face_id = edge->pair->face->id;
                // Real diff = distance_to_cluster(face_offsets[pair_face_id], cluster_centers[cid], pair_face_id);
                Real diff = distance_to_cluster(pair_face_id, cluster_center_face_ids[cid], type);

                if (unclustered_face_diff_distance[pair_face_id] > diff)
                {
                    auto backup_cid = face_cluster_ids[pair_face_id];
                    face_cluster_ids[pair_face_id] = cid;
                    if (has_nonmanifold_vertex(pair_face_id))
                    {
                        face_cluster_ids[pair_face_id] = backup_cid;
                    }
                    else
                    {
                        unclustered_face_diff_distance[pair_face_id] = diff;
                    }
                }
            }
            edge = edge->next;
        } while (edge != face->edge);
    }
    for (auto i = 0; i < nf; i++)
    {
        if (unclustered_face_tags[i] && face_cluster_ids[i] > 0)
        {
            unclustered_faces.push(std::make_tuple(m_pmesh->get_face(i), unclustered_face_diff_distance[i], face_cluster_ids[i]));
        }
    }

    while (!unclustered_faces.empty())
    {
        auto entry = unclustered_faces.top();
        auto face = std::get<0>(entry);
        auto diff = std::get<1>(entry);
        auto cid = std::get<2>(entry);
        unclustered_faces.pop();

        if (face_cluster_ids[face->id] != cid)
            continue;

        auto edge = face->edge;
        do
        {
            if (edge->pair->face)
            {
                const auto &pair_face_id = edge->pair->face->id;
                if (unclustered_face_tags[pair_face_id] && face_cluster_ids[pair_face_id] != cid && (!check_sharp_feature || color_feature_edge_tags[edge->id] == false))
                {
                    Real diff2 = distance_to_cluster(pair_face_id, cluster_center_face_ids[cid], type);
                    if (diff2 < unclustered_face_diff_distance[pair_face_id])
                    {
                        auto backup_cid = face_cluster_ids[pair_face_id];
                        face_cluster_ids[pair_face_id] = cid;
                        if (has_nonmanifold_vertex(pair_face_id))
                        {
                            face_cluster_ids[pair_face_id] = backup_cid;
                        }
                        else
                        {
                            unclustered_face_diff_distance[pair_face_id] = diff2;
                            unclustered_faces.push(std::make_tuple(edge->pair->face, diff2, cid));
                        }
                    }
                }
            }
            edge = edge->next;

        } while (edge != face->edge);
    }
    process_unlabeled_faces();
}
//////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::face_clustering_via_seeds(const FaceClusterType type, const std::vector<FaceClusterType> &seed_face_tags)
{
    auto nf = m_pmesh->get_num_of_faces();
    const std::vector<Real> &face_colors = (type == FaceClusterType::FC_CDF) ? face_cdf_colors : face_dcdf_colors;
    const std::vector<TinyVector<Real, 3>> &offsets = (type == FaceClusterType::FC_CDF) ? coffset : doffset;
    face_cluster_ids.assign(nf, 0); // 0: unlabeled; others: label_id
    std::unordered_set<ptrdiff_t> fixed_face_set;
    cluster_center_face_ids.resize(1, -1);
    int cluster_count = 1;

    bool check_sharp_feature = (type == FaceClusterType::FC_CDF);

    for (ptrdiff_t i = 0; i < nf; i++)
    {
        if (seed_face_tags[i] == type)
        {
            face_cluster_ids[i] = cluster_count++;
            cluster_center_face_ids.push_back(i);
            fixed_face_set.insert(i);
        }
        else if (face_colors[i] <= confuse_band)
        {
            fixed_face_set.insert(i);
        }
    }

    typedef std::tuple<MeshLib::HE_face<Real> *, Real, ptrdiff_t> UnClusterFace;
    class unclustered_face_compare
    {
    public:
        int operator()(const UnClusterFace &a, const UnClusterFace &b)
        {
            return std::get<1>(a) > std::get<1>(b);
        }
    };
    std::priority_queue<UnClusterFace, std::deque<UnClusterFace>, unclustered_face_compare> unclustered_faces;
    std::vector<Real> unclustered_face_diff_distance(nf, std::numeric_limits<Real>::max());

    for (auto i = 0; i < nf; i++)
    {
        if (seed_face_tags[i] != type)
            continue;

        auto face = m_pmesh->get_face(i);
        auto edge = face->edge;
        auto cid = face_cluster_ids[face->id];
        do
        {
            if (edge->pair->face && (!check_sharp_feature || color_feature_edge_tags[edge->id] == false))
            {
                auto pair_face_id = edge->pair->face->id;
                if (fixed_face_set.find(pair_face_id) == fixed_face_set.end())
                {
                    // Real diff = distance_to_cluster(offsets[pair_face_id], cluster_centers[cid], pair_face_id);
                    Real diff = distance_to_cluster(pair_face_id, cluster_center_face_ids[cid], type);

                    if (unclustered_face_diff_distance[pair_face_id] > diff)
                    {
                        auto backup_cid = face_cluster_ids[pair_face_id];
                        face_cluster_ids[pair_face_id] = cid;
                        if (has_nonmanifold_vertex(pair_face_id))
                        {
                            face_cluster_ids[pair_face_id] = backup_cid;
                        }
                        else
                        {
                            unclustered_face_diff_distance[pair_face_id] = diff;
                        }
                    }
                }
            }
            edge = edge->next;
        } while (edge != face->edge);
    }
    for (auto i = 0; i < nf; i++)
    {
        if (fixed_face_set.find(i) == fixed_face_set.end() && face_cluster_ids[i] > 0)
        {
            unclustered_faces.push(std::make_tuple(m_pmesh->get_face(i), unclustered_face_diff_distance[i], face_cluster_ids[i]));
        }
    }

    while (!unclustered_faces.empty())
    {
        auto entry = unclustered_faces.top();
        auto face = std::get<0>(entry);
        auto diff = std::get<1>(entry);
        auto cid = std::get<2>(entry);
        unclustered_faces.pop();

        if (face_cluster_ids[face->id] != cid)
            continue;

        auto edge = face->edge;
        do
        {
            if (edge->pair->face && (!check_sharp_feature || color_feature_edge_tags[edge->id] == false))
            {
                const auto &pair_face_id = edge->pair->face->id;
                if (fixed_face_set.find(pair_face_id) == fixed_face_set.end() && face_cluster_ids[pair_face_id] != cid)
                {
                    // Real diff2 = distance_to_cluster(offsets[pair_face_id], cluster_centers[cid], pair_face_id);
                    Real diff2 = distance_to_cluster(pair_face_id, cluster_center_face_ids[cid], type);

                    if (diff2 < unclustered_face_diff_distance[pair_face_id])
                    {
                        auto backup_cid = face_cluster_ids[pair_face_id];
                        face_cluster_ids[pair_face_id] = cid;
                        if (has_nonmanifold_vertex(pair_face_id))
                        {
                            face_cluster_ids[pair_face_id] = backup_cid;
                        }
                        else
                        {
                            unclustered_face_diff_distance[pair_face_id] = diff2;
                            unclustered_faces.push(std::make_tuple(edge->pair->face, diff2, cid));
                        }
                    }
                }
            }
            edge = edge->next;

        } while (edge != face->edge);
    }

    // mark the potential holes
    std::queue<MeshLib::HE_face<Real> *> face_queue;
    std::vector<bool> visited_faces(nf, false);
    for (ptrdiff_t i = 0; i < nf; i++)
    {
        if (seed_face_tags[i] == type)
        {
            face_queue.push(m_pmesh->get_face(i));
        }
    }
    while (!face_queue.empty())
    {
        auto face = face_queue.front();
        face_queue.pop();
        if (visited_faces[face->id])
            continue;
        visited_faces[face->id] = true;
        auto edge = face->edge;
        do
        {
            if (edge->pair->face && !visited_faces[edge->pair->face->id] && face_cluster_ids[edge->pair->face->id] == face_cluster_ids[face->id])
            {
                face_queue.push(edge->pair->face);
            }
            edge = edge->next;
        } while (edge != face->edge);
    }
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nf; i++)
    {
        if (!visited_faces[i])
        {
            face_cluster_ids[i] = 0;
        }
    }

    merge_clusters(type);
}
//////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::process_unlabeled_faces()
{
    std::vector<ptrdiff_t> unprocessed_faces, remain_unprocessed_faces;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        if (face_cluster_ids[i] == 0)
            unprocessed_faces.push_back(i);
    }
    bool processed = false;
    do
    {
        processed = false;
        remain_unprocessed_faces.resize(0);
        for (auto i : unprocessed_faces)
        {
            if (face_cluster_ids[i] > 0)
                continue;
            auto face = m_pmesh->get_face(i);
            auto edge = face->edge;
            bool corrected = false;
            std::unordered_map<ptrdiff_t, int> neighbor_cluster_count;
            do
            {
                if (edge->pair->face && face_cluster_ids[edge->pair->face->id] > 0)
                {
                    neighbor_cluster_count[face_cluster_ids[edge->pair->face->id]]++;
                }
                edge = edge->next;
            } while (edge != face->edge);
            if (neighbor_cluster_count.size() == 1)
            {
                face_cluster_ids[i] = neighbor_cluster_count.begin()->first;
                corrected = true;
            }
            else if (neighbor_cluster_count.size() > 1)
            {
                for (const auto &entry : neighbor_cluster_count)
                {
                    if (entry.second >= 2)
                    {
                        face_cluster_ids[i] = entry.first;
                        corrected = true;
                        break;
                    }
                }
            }
            if (!corrected)
            {
                remain_unprocessed_faces.push_back(i);
            }
        }
        unprocessed_faces = remain_unprocessed_faces;
    } while (processed);

    for (auto i : unprocessed_faces)
    {
        auto hf = m_pmesh->get_face(i);
        auto he = hf->edge;
        do
        {
            if (he->pair->face && face_cluster_ids[he->pair->face->id] > 0)
            {
                face_cluster_ids[i] = face_cluster_ids[he->pair->face->id];
                break;
            }
            he = he->next;
        } while (he != hf->edge);
    }
}
//////////////////////////////////
template <typename Real>
int QCDF2QuadMesh<Real>::cluster_via_polarization(const FaceClusterType type)
{
    auto nf = m_pmesh->get_num_of_faces();
    std::vector<FaceClusterType> face_cluster_types(nf, FaceClusterType::FC_UNDEFINED);
    int num_seeds = identify_seed_faces(type, face_cluster_types);
    // cluster_centers.reserve(num_seeds + 1);
    face_clustering_via_seeds(type, face_cluster_types);
    return num_seeds + 1; // 1 for the unclustered faces
}
//////////////////////////////////
template <typename Real>
bool QCDF2QuadMesh<Real>::handle_incorrect_clusters(const FaceClusterType type, int &num_clusters)
{
    std::vector<std::unordered_set<ptrdiff_t>> cluster_adjacency(num_clusters);
    for (auto i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge < edge->pair)
        {
            if (edge->face && edge->pair->face == 0)
            {
                auto cid0 = face_cluster_ids[edge->face->id];
                if (cid0 > 0)
                    cluster_adjacency[cid0].insert(-1); // use -1 to indicate the boundary
            }
            else if (edge->face == 0 && edge->pair->face)
            {
                auto cid1 = face_cluster_ids[edge->pair->face->id];
                if (cid1 > 0)
                    cluster_adjacency[cid1].insert(-1);
            }
            else if (edge->face && edge->pair->face)
            {
                auto cid0 = face_cluster_ids[edge->face->id], cid1 = face_cluster_ids[edge->pair->face->id];
                if (cid0 != cid1 && cid0 > 0 && cid1 > 0)
                {
                    cluster_adjacency[cid0].insert(cid1);
                    cluster_adjacency[cid1].insert(cid0);
                }
            }
        }
    }
    std::unordered_set<ptrdiff_t> invalid_clusters;
    std::vector<bool> invalid_clusters_tags(num_clusters, false);
    invalid_clusters_tags[0] = true;
    bool encountered_invalid = false;

    for (auto i = 1; i < num_clusters; i++)
    {
        if (!cluster_adjacency[i].empty() && cluster_adjacency[i].size() < 3)
        {
            invalid_clusters_tags[i] = true;
            encountered_invalid = true;
        }
    }

    if (encountered_invalid)
    {
        auto nf = m_pmesh->get_num_of_faces();
#pragma omp parallel for
        for (auto i = 0; i < nf; i++)
        {
            if (invalid_clusters_tags[face_cluster_ids[i]])
            {
                face_cluster_ids[i] = 0;
            }
        }
        process_uncluster_faces(type, num_clusters);
    }

    return encountered_invalid;
}
//////////////////////////////////
template <typename Real>
int QCDF2QuadMesh<Real>::reindex_clusters()
{
    std::vector<bool> valid_cluster_tags(cluster_center_face_ids.size(), false);
    std::vector<ptrdiff_t> reindex_map(cluster_center_face_ids.size(), -1);
    int counter = 0;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto cid = face_cluster_ids[i];
        if (cid > 0)
            valid_cluster_tags[cid] = true;
    }
    for (ptrdiff_t i = 0; i < (ptrdiff_t)valid_cluster_tags.size(); i++)
    {
        if (valid_cluster_tags[i])
            reindex_map[i] = counter++;
    }
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        if (face_cluster_ids[i] > 0)
            face_cluster_ids[i] = reindex_map[face_cluster_ids[i]];
        else
            face_cluster_ids[i] = -1;
    }
    std::vector<ptrdiff_t> cluster_center_face_ids_new(counter);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)valid_cluster_tags.size(); i++)
    {
        if (valid_cluster_tags[i])
        {
            auto new_id = reindex_map[i];
            cluster_center_face_ids_new[new_id] = cluster_center_face_ids[i];
        }
    }
    cluster_center_face_ids = cluster_center_face_ids_new;

    return counter;
}
///////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::dcdf_build_quad_patches()
{
    std::vector<std::vector<size_t>> patches;
    std::vector<size_t> poly;
    std::unordered_set<ptrdiff_t> visited_clusters;
    quad_vertices.assign(cluster_center_face_ids.size(), TinyVector<Real, 3>(0, 0, 0));

    for (size_t i = 1; i < cluster_center_face_ids.size(); i++)
    {
        if (pattern_subdivision)
            quad_vertices[i] = face_centroids[cluster_center_face_ids[i]];
        else
            quad_vertices[i] = face_centroids[cluster_center_face_ids[i]] + doffset[cluster_center_face_ids[i]];
    }
    // quad_vertices.assign(cluster_centers.begin(), cluster_centers.end());
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        poly.resize(0), visited_clusters.clear();
        visited_clusters.insert(-1);
        auto hv = m_pmesh->get_vertex(i);
        auto he = hv->edge;
        do
        {
            if (he->face && visited_clusters.find(face_cluster_ids[he->face->id]) == visited_clusters.end())
            {
                visited_clusters.insert(face_cluster_ids[he->face->id]);
                poly.emplace_back(face_cluster_ids[he->face->id]);
            }
            he = he->pair->next;
        } while (he != hv->edge);
        if (poly.size() < 3)
            continue;
        std::reverse(poly.begin(), poly.end());
        patches.push_back(poly);
    }

    // if (m_debug_mode)
    {
        SavePLYMesh_with_color(path_concatenate(debug_dir, "dcdf_polygon_output.ply"), quad_vertices, patches);
    }

    class edgeinfo
    {
    public:
        edgeinfo()
        {
            fid[0] = fid[1] = -1;
            edge_vid[0] = edge_vid[1] = -1;
            opposite_vid[0] = opposite_vid[1] = -1;
            similarity_score = std::numeric_limits<float>::max();
        }
        ptrdiff_t fid[2];
        ptrdiff_t edge_vid[2];
        ptrdiff_t opposite_vid[2];
        float similarity_score;
        bool operator<(const edgeinfo &other) const
        {
            return similarity_score < other.similarity_score;
        }
    };

    std::vector<int> face_tags(patches.size(), false);
    std::unordered_map<MySortedTuple<ptrdiff_t, 2, true>, edgeinfo> edge_to_opposite_map;
    for (ptrdiff_t fid = 0; fid < (ptrdiff_t)patches.size(); fid++)
    {
        if (patches[fid].size() != 3)
            continue;
        for (int i = 0; i < 3; i++)
        {
            auto vi = patches[fid][i], vii = patches[fid][(i + 1) % 3];
            MySortedTuple<ptrdiff_t, 2, true> edge(patches[fid][i], patches[fid][(i + 1) % 3]);
            int order = vi < vii ? 0 : 1;
            auto iter = edge_to_opposite_map.find(edge);
            if (iter == edge_to_opposite_map.end())
            {
                edgeinfo tmp_edgeinfo;
                tmp_edgeinfo.edge_vid[0] = edge[0];
                tmp_edgeinfo.edge_vid[1] = edge[1];
                tmp_edgeinfo.fid[order] = fid;
                tmp_edgeinfo.opposite_vid[order] = patches[fid][(i + 2) % 3];
                edge_to_opposite_map[edge] = tmp_edgeinfo;
            }
            else
            {
                iter->second.fid[order] = fid;
                iter->second.opposite_vid[order] = patches[fid][(i + 2) % 3];
            }
        }
    }
    std::unordered_map<ptrdiff_t, Real> cluster_boundary_length;
    std::unordered_map<MySortedTuple<ptrdiff_t, 2, true>, Real> shared_boundary_lengths;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i += 2)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge->face && edge->pair->face)
        {
            auto cid0 = face_cluster_ids[edge->face->id];
            auto cid1 = face_cluster_ids[edge->pair->face->id];
            if (cid0 != cid1)
            {
                MySortedTuple<ptrdiff_t, 2, true> edge_key(cid0, cid1);
                auto len = edge->GetLength();
                shared_boundary_lengths[edge_key] += len;
                cluster_boundary_length[cid0] += len;
                cluster_boundary_length[cid1] += len;
            }
        }
    }

    std::vector<edgeinfo> candidate_edges;
    for (auto &item : edge_to_opposite_map)
    {
        if (item.second.fid[0] >= 0 && item.second.fid[1] >= 0)
        {
            auto min_cluster_boundary = std::min(cluster_boundary_length[item.first[0]], cluster_boundary_length[item.first[1]]) + (Real)1.0e-10;
            item.second.similarity_score = (float)shared_boundary_lengths[item.first];
            candidate_edges.push_back(item.second);
        }
    }
    std::sort(candidate_edges.begin(), candidate_edges.end());
    quad_faces.resize(0), quad_faces.reserve(patches.size());
    for (auto &edge : candidate_edges)
    {
        if (face_tags[edge.fid[0]] || face_tags[edge.fid[1]])
            continue;
        face_tags[edge.fid[0]] = true;
        face_tags[edge.fid[1]] = true;
        std::vector<size_t> quad(4);
        quad[0] = (size_t)edge.opposite_vid[0];
        quad[1] = (size_t)edge.edge_vid[0];
        quad[2] = (size_t)edge.opposite_vid[1];
        quad[3] = (size_t)edge.edge_vid[1];
        quad_faces.push_back(quad);
    }
    for (size_t fid = 0; fid < patches.size(); fid++)
    {
        if (!face_tags[fid])
        {
            quad_faces.push_back(patches[fid]);
        }
    }

    // remove unused vertices and reindex, and update other data structures
    std::vector<bool> vertex_reference_tag(quad_vertices.size(), false);
    std::vector<ptrdiff_t> vertex_new_ids(quad_vertices.size(), -1);
    for (size_t i = 0; i < quad_faces.size(); i++)
    {
        for (size_t j = 0; j < quad_faces[i].size(); j++)
        {
            vertex_reference_tag[quad_faces[i][j]] = true;
        }
    }
    std::vector<TinyVector<Real, 3>> new_quad_vertices;
    new_quad_vertices.reserve(quad_vertices.size());
    int non_quad_num = 0;
    for (size_t i = 0; i < quad_faces.size(); i++)
    {
        if (quad_faces[i].size() != 4)
            non_quad_num++;
    }

    size_t count = 0;
    for (size_t i = 0; i < quad_vertices.size(); i++)
    {
        if (vertex_reference_tag[i])
        {
            vertex_new_ids[i] = count++;
            new_quad_vertices.emplace_back(quad_vertices[i]);
        }
    }
    bool has_unreferenced_vertex = count < quad_vertices.size();

    if (has_unreferenced_vertex)
    {
        quad_vertices = new_quad_vertices;
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < (ptrdiff_t)quad_faces.size(); i++)
        {
            for (size_t j = 0; j < quad_faces[i].size(); j++)
            {
                quad_faces[i][j] = vertex_new_ids[quad_faces[i][j]];
            }
        }
    }
    if (non_quad_num > 0)
    {
        if (m_verbose)
            std::cout << "Warning: " << non_quad_num << " non-quad polygons exist in the extracted quad mesh!" << std::endl;
    }
    bool nonmanifold_issue = has_nonmanifold_issue(quad_vertices.size(), quad_faces);
    if (nonmanifold_issue)
    {
        if (m_verbose)
            std::cout << "Warning: Non-manifold vertices/edges exist in the extracted quad mesh!" << std::endl;
    }
    else
    {
        if (non_quad_num > 0 && improve_mode)
        {
            non_quad_num = fix_nonquad_faces(non_quad_num);
            if (m_verbose)
            {
                if (non_quad_num > 0)
                    std::cout << "Warning: " << non_quad_num << " non-quad polygons still exist in the extracted quad mesh after fixing!" << std::endl;
                else
                    std::cout << "All non-quad polygons are successfully fixed!" << std::endl;
            }
        }
    }
}
///////////////////////////////////
template <typename Real>
bool QCDF2QuadMesh<Real>::identify_boundary_dcdf_vertex(const std::vector<Real> &boundary_dcdf_values, const ptrdiff_t curve_point_index)
{
    const auto &current_value = boundary_dcdf_values[curve_point_index];
    for (int index = -ring_size; index <= ring_size; index++)
    {
        if (index == 0)
            continue;
        ptrdiff_t neighbor_index = curve_point_index + index;
        while (neighbor_index < 0)
            neighbor_index += boundary_dcdf_values.size();
        neighbor_index = neighbor_index % boundary_dcdf_values.size();
        if (boundary_dcdf_values[neighbor_index] > current_value)
        {
            return false;
        }
    }
    return true;
}
///////////////////////////////////
template <typename Real>
TinyVector<Real, 3> QCDF2QuadMesh<Real>::compute_polyface_normal(const std::vector<TinyVector<Real, 3>> &face_vertices)
{
    TinyVector<Real, 3> normal(0, 0, 0);
    size_t nv = face_vertices.size();
    for (size_t i = 0; i < nv; i++)
    {
        const auto &v0 = face_vertices[i];
        const auto &v1 = face_vertices[(i + 1) % nv];
        normal += v0.Cross(v1);
    }
    normal.Normalize();
    return normal;
}
///////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::cdf_build_quad_patches()
{
    std::vector<int> vertex_tags(m_pmesh->get_num_of_vertices(), 0);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto hv = m_pmesh->get_vertex(i);
        auto he = hv->edge;
        std::unordered_set<ptrdiff_t> cluster_set;
        bool is_boundary_vertex = false;
        do
        {
            if (he->face)
                cluster_set.insert(face_cluster_ids[he->face->id]);
            else
            {
                cluster_set.insert(-1);
                is_boundary_vertex = true;
            }
            he = he->pair->next;
        } while (he != hv->edge);
        if (cluster_set.size() >= 3 || (is_boundary_vertex && vertex_feature_tag[i]))
            vertex_tags[i] = 1;
    }
    /////////////////////////////////////////////
    std::vector<int> quad_vertex_type; // -1: interior; others: the i-th boundary curve
    std::unordered_map<ptrdiff_t, int> vertex_to_boundary_curve_id;
    for (size_t i = 0; i < m_pmesh->boundaryvertices.size(); i++)
    {
        for (size_t j = 0; j < m_pmesh->boundaryvertices[i].size(); j++)
        {
            auto hv = m_pmesh->boundaryvertices[i][j];
            if (vertex_tags[hv->id] == 1)
            {
                vertex_to_boundary_curve_id[hv->id] = (int)i;
            }
        }
    }

    std::vector<ptrdiff_t> vertex_new_ids(m_pmesh->get_num_of_vertices(), -1);
    ptrdiff_t vcount = 0;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        if (vertex_tags[i] == 1)
        {
            vertex_new_ids[i] = vcount++;

            auto iter = vertex_to_boundary_curve_id.find(i);
            if (iter == vertex_to_boundary_curve_id.end())
            {
                quad_vertex_type.push_back(-1);
            }
            else
            {
                quad_vertex_type.push_back(iter->second);
            }
        }
    }
    quad_vertices.reserve(vcount);
    quad_vertices.resize(0);
    quad_faces.resize(0);
    m_pmesh->reset_edges_tag(false);
    std::vector<size_t> poly_vertex_ids;
    std::vector<ptrdiff_t> corresponding_mesh_vertex_ids;
    for (ptrdiff_t vi = 0; vi < m_pmesh->get_num_of_vertices(); vi++)
    {
        if (vertex_tags[vi] == 0)
            continue;
        quad_vertices.emplace_back(m_pmesh->get_vertex(vi)->pos);
        corresponding_mesh_vertex_ids.push_back(vi);

        auto vert = m_pmesh->get_vertex(vi);
        auto vedge = vert->edge;
        do
        {
            auto edge = vedge;
            if (edge->tag || !edge->face ||
                (edge->pair->face && face_cluster_ids[edge->face->id] == face_cluster_ids[edge->pair->face->id]))
                ;
            else
            {
                bool is_boundary = (edge->pair->face == 0);
                poly_vertex_ids.resize(0);
                do
                {
                    if (vertex_new_ids[edge->pair->vert->id] >= 0)
                    {
                        poly_vertex_ids.push_back((size_t)vertex_new_ids[edge->pair->vert->id]);
                    }
                    edge->tag = true;
                    auto next_edge = edge->next;
                    do
                    {
                        if (!next_edge->pair->face ||
                            (face_cluster_ids[next_edge->face->id] != face_cluster_ids[next_edge->pair->face->id]))
                        {
                            break;
                        }
                        else
                        {
                            next_edge = next_edge->pair->next;
                        }
                    } while (1);
                    edge = next_edge;
                } while (edge->tag == false);
                if (poly_vertex_ids.size() >= 3)
                    quad_faces.emplace_back(poly_vertex_ids);
            }

            vedge = vedge->pair->next;
        } while (vedge != vert->edge);
    }

    // remove unused vertices and reindex, and update other data structures
    std::vector<bool> vertex_reference_tag(quad_vertices.size(), false);
    std::vector<ptrdiff_t> vertex_new_ids2(quad_vertices.size(), -1);
    for (size_t i = 0; i < quad_faces.size(); i++)
    {
        for (size_t j = 0; j < quad_faces[i].size(); j++)
        {
            vertex_reference_tag[quad_faces[i][j]] = true;
        }
    }
    std::vector<TinyVector<Real, 3>> new_quad_vertices;
    new_quad_vertices.reserve(quad_vertices.size());
    size_t count = 0;
    for (size_t i = 0; i < quad_vertices.size(); i++)
    {
        if (vertex_reference_tag[i])
        {
            vertex_new_ids2[i] = count++;
            new_quad_vertices.emplace_back(quad_vertices[i]);
        }
    }
    bool has_unreferenced_vertex = count < quad_vertices.size();
    if (has_unreferenced_vertex)
    {
        quad_vertices = new_quad_vertices;
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < (ptrdiff_t)quad_faces.size(); i++)
        {
            for (size_t j = 0; j < quad_faces[i].size(); j++)
            {
                quad_faces[i][j] = vertex_new_ids2[quad_faces[i][j]];
            }
        }
        std::vector<int> new_quad_vertex_type;
        new_quad_vertex_type.reserve(quad_vertices.size());
        for (size_t i = 0; i < quad_vertex_type.size(); i++)
        {
            if (vertex_reference_tag[i])
            {
                new_quad_vertex_type.push_back(quad_vertex_type[i]);
            }
        }
        quad_vertex_type = new_quad_vertex_type;
    }

    // if (m_debug_mode)
    {
        std::string name = input_method_type == FaceClusterType::FC_CDF ? "cdf_polygon_output.ply" : "cdfdcdf_polygon_output.ply";
        SavePLYMesh_with_color(path_concatenate(debug_dir, name), quad_vertices, quad_faces);
    }

    bool nonmanifold_issue = has_nonmanifold_issue(quad_vertices.size(), quad_faces);
    if (nonmanifold_issue)
    {
        if (m_verbose)
            std::cout << "Warning: Non-manifold vertices/edges exist in the extracted quad mesh!" << std::endl;
        return;
    }

    // collaspe short edges to achieve more quads with OpenMesh library
    // Create polymesh
    typedef OpenMesh::PolyMesh_ArrayKernelT<> PolyMesh;
    PolyMesh polymesh;
    polymesh.request_vertex_status();
    polymesh.request_edge_status();
    polymesh.request_face_status();

    OpenMesh::VPropHandleT<int> v_feature_tag;
    OpenMesh::VPropHandleT<size_t> v_id_tag;
    OpenMesh::FPropHandleT<float> f_max_face_edge_length;
    polymesh.add_property(v_id_tag, "v:id_tag");
    polymesh.add_property(v_feature_tag, "v:feature_tag");
    polymesh.add_property(f_max_face_edge_length, "f:max_face_edge_length");

    for (size_t v = 0; v < quad_vertices.size(); v++)
    {
        auto vh = polymesh.add_vertex(OpenMesh::Vec3f((float)quad_vertices[v][0], (float)quad_vertices[v][1], (float)quad_vertices[v][2]));
        polymesh.property(v_feature_tag, vh) = quad_vertex_type[v];
        polymesh.property(v_id_tag, vh) = v;
    }

    std::vector<PolyMesh::VertexHandle> face_vertices;
    face_vertices.reserve(8);
    for (const auto &f : quad_faces)
    {
        face_vertices.resize(0);
        for (const auto &vid : f)
        {
            face_vertices.push_back(PolyMesh::VertexHandle((int)vid));
        }
        polymesh.add_face(face_vertices);
    }
    const Real cos_angle_threshold = std::cos(edge_collapse_normal_threshold * M_PI / 180);
    std::unordered_map<ptrdiff_t, ptrdiff_t> vertex_collapse_map;

    // Initialize edge collapse priority queue

    class EdgeCollapseInfo
    {
    public:
        EdgeCollapseInfo(PolyMesh &polymesh = PolyMesh(), PolyMesh::EdgeHandle eh = PolyMesh::EdgeHandle())
            : edge_handle(eh), edge_length(0), edge_type(1000), vertex_type(1000), min_face_degree(0), max_face_degree(0)
        {
            update_info(polymesh);
        }

        bool
        operator>(const EdgeCollapseInfo &other) const
        {
            if (edge_type > other.edge_type)
                return true;
            else if (edge_type < other.edge_type)
                return false;
            else if (vertex_type > other.vertex_type)
                return true;
            if (edge_length > other.edge_length)
                return true;
            return false;
        }

        void update_info(PolyMesh &polymesh)
        {
            if (edge_handle.is_valid() == false)
                return;
            auto heh = polymesh.halfedge_handle(edge_handle, 0);
            const auto &vh0 = polymesh.from_vertex_handle(heh);
            const auto &vh1 = polymesh.to_vertex_handle(heh);
            edge_length = (polymesh.point(vh0) - polymesh.point(vh1)).length();
            // get adjacent faces
            auto fh0 = polymesh.face_handle(heh);
            auto fh1 = polymesh.opposite_face_handle(heh);
            std::vector<int> face_degrees;
            if (fh0.is_valid())
            {
                face_degrees.push_back(polymesh.valence(fh0));
            }
            if (fh1.is_valid())
            {
                face_degrees.push_back(polymesh.valence(fh1));
            }
            min_face_degree = *std::min_element(face_degrees.begin(), face_degrees.end());
            max_face_degree = *std::max_element(face_degrees.begin(), face_degrees.end());
            if (min_face_degree > 4)
                edge_type = 0;
            else if (min_face_degree == 3 && max_face_degree > 4)
                edge_type = 1;
            else if (min_face_degree == 4 && max_face_degree > 4)
                edge_type = 2;
            else if (min_face_degree == 3 && max_face_degree == 3)
                edge_type = 3;
            else
                edge_type = 4;
            vertex_type = std::abs((int)polymesh.valence(vh0) - 3) + std::abs((int)polymesh.valence(vh1) - 3);
        }

    public:
        float edge_length;
        int edge_type, vertex_type;
        PolyMesh::EdgeHandle edge_handle;
        int min_face_degree, max_face_degree;
    };

    const int edge_type_threshold = 3; // only collapse edge with type less than this value

    std::priority_queue<EdgeCollapseInfo, std::vector<EdgeCollapseInfo>, std::greater<EdgeCollapseInfo>>
        edge_queue;

    float min_edge_length = std::numeric_limits<float>::max();
    for (const auto &edge : polymesh.edges())
    {

        EdgeCollapseInfo edge_info(polymesh, edge);

        min_edge_length = std::min(min_edge_length, edge_info.edge_length);

        auto heh = polymesh.halfedge_handle(edge, 0);
        auto fh0 = polymesh.face_handle(heh);
        auto fh1 = polymesh.opposite_face_handle(heh);
        if (fh0.is_valid())
        {
            polymesh.property(f_max_face_edge_length, fh0) =
                std::max(polymesh.property(f_max_face_edge_length, fh0), edge_info.edge_length);
        }
        if (fh1.is_valid())
        {
            polymesh.property(f_max_face_edge_length, fh1) =
                std::max(polymesh.property(f_max_face_edge_length, fh1), edge_info.edge_length);
        }

        if (edge_info.edge_type <= edge_type_threshold)
            edge_queue.push(edge_info);
    }

    while (!edge_queue.empty())
    {
        auto edge_info = edge_queue.top();
        edge_queue.pop();

        if (edge_info.edge_handle.is_valid() == false)
            continue;

        auto heh = polymesh.halfedge_handle(edge_info.edge_handle, 0);
        auto vh_from = polymesh.from_vertex_handle(heh);
        auto vh_to = polymesh.to_vertex_handle(heh);
        auto vh_from_feature_type = polymesh.property(v_feature_tag, vh_from);
        auto vh_to_feature_type = polymesh.property(v_feature_tag, vh_to);

        // do not collapse edge between different feature types
        if (vh_from_feature_type >= 0 && vh_to_feature_type >= 0 && vh_from_feature_type != vh_to_feature_type)
            continue;

        int valence_from = (int)polymesh.valence(vh_from);
        int valence_to = (int)polymesh.valence(vh_to);

        if (valence_to < valence_from)
        {
            std::swap(vh_from, vh_to);
            std::swap(valence_from, valence_to);
            std::swap(vh_from_feature_type, vh_to_feature_type);
            heh = polymesh.opposite_halfedge_handle(heh);
        }
        // only collapse boundary edge if one vertex has valence 2
        if (polymesh.is_boundary(heh) && valence_from != 2)
            continue;

        // Check if collapse is legal
        if (!polymesh.is_collapse_ok(heh))
            continue;

        edge_info.update_info(polymesh);
        if (edge_info.edge_type > edge_type_threshold)
            continue;

        auto fh0 = polymesh.face_handle(heh);
        auto fh1 = polymesh.opposite_face_handle(heh);
        float max_face_edge_length = 0;
        if (fh0.is_valid())
        {
            max_face_edge_length = polymesh.property(f_max_face_edge_length, fh0);
        }
        if (fh1.is_valid())
        {
            max_face_edge_length = std::max(max_face_edge_length, polymesh.property(f_max_face_edge_length, fh1));
        }

        // if (edge_info.edge_type == 2 && edge_info.edge_length > edge_collapse_ratio * max_face_edge_length)
        if (edge_info.edge_type > 0 && edge_info.edge_type != 3 && edge_info.edge_length > edge_collapse_ratio * max_face_edge_length)
            // if (edge_info.edge_length > edge_collapse_ratio * max_face_edge_length)
            continue;

        auto edge_is_boundary = polymesh.is_boundary(heh);
        auto vh_from_valence = polymesh.valence(vh_from);
        auto vh_to_valence = polymesh.valence(vh_to);
        auto min_valence = std::min(vh_from_valence, vh_to_valence);
        auto max_valence = std::max(vh_from_valence, vh_to_valence);
        if (edge_is_boundary)
        {
            if (min_valence != 2)
                continue;
        }
        else
        {
            if (edge_info.edge_type = 0 && max_valence > 4)
                continue;
            if (edge_info.edge_type > 0 && (min_valence != 3 || max_valence != 3))
                continue;
            if (edge_info.edge_type > 1 && edge_info.edge_type != 3 && max_valence > 3)
                continue;
            if (edge_info.edge_type == 3 && max_valence > 4)
                continue;
        }
        // if (((!edge_is_boundary && min_valence != 3) || (edge_is_boundary && min_valence != 2)))
        //     continue;

        auto vh_from_idx = polymesh.property(v_id_tag, vh_from);
        auto vh_to_idx = polymesh.property(v_id_tag, vh_to);

        if (
            tri_mesh_vertex_cdfdcdf_types[corresponding_mesh_vertex_ids[vh_from_idx]] >= 0 &&
            tri_mesh_vertex_cdfdcdf_types[corresponding_mesh_vertex_ids[vh_to_idx]] >= 0 &&
            tri_mesh_vertex_cdfdcdf_types[corresponding_mesh_vertex_ids[vh_from_idx]] != tri_mesh_vertex_cdfdcdf_types[corresponding_mesh_vertex_ids[vh_to_idx]])
            continue;

        // don't move boundary vertex to interior vertex
        bool should_swap = false, swaped = false;
        if ((vh_from_feature_type != -1 && vh_to_feature_type == -1) ||
            (vertex_feature_tag[corresponding_mesh_vertex_ids[vh_from_idx]] && !vertex_feature_tag[corresponding_mesh_vertex_ids[vh_to_idx]]))
        {

            should_swap = true;
            if (valence_from != 2)
            {
                std::swap(vh_from, vh_to);
                std::swap(vh_from_idx, vh_to_idx);
                std::swap(valence_from, valence_to);
                heh = polymesh.opposite_halfedge_handle(heh);
                swaped = true;
            }
        }

        bool normal_deviation_exceeded = false;
        std::vector<TinyVector<Real, 3>> before_face_points, after_face_points;
        before_face_points.reserve(8);
        after_face_points.reserve(8);
        for (auto vf_it = polymesh.cvf_iter(vh_from); vf_it.is_valid(); ++vf_it)
        {
            auto fh = *vf_it;
            bool is_tri = polymesh.valence(fh) == 3;
            bool contains_vh_to = false;
            before_face_points.resize(0), after_face_points.resize(0);
            for (auto fv_it = polymesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it)
            {
                auto p = polymesh.point(*fv_it);
                auto pv = TinyVector<Real, 3>((Real)p[0], (Real)p[1], (Real)p[2]);
                before_face_points.emplace_back(pv);
                if (*fv_it == vh_from)
                {
                    auto p = polymesh.point(vh_to);
                    after_face_points.push_back(TinyVector<Real, 3>((Real)p[0], (Real)p[1], (Real)p[2]));
                }
                else
                {
                    after_face_points.push_back(pv);
                }
                if (*fv_it == vh_to)
                {
                    contains_vh_to = true;
                }
            }

            if (is_tri && contains_vh_to)
                continue; // collapsing a vertex of a triangle face to another vertex of the same face will remove the face, skip normal check

            auto before_normal = compute_polyface_normal(before_face_points);
            auto after_normal = compute_polyface_normal(after_face_points);
            Real dot_product = before_normal.Dot(after_normal);
            if (dot_product < cos_angle_threshold)
            {
                normal_deviation_exceeded = true;
                break;
            }
        }

        if (normal_deviation_exceeded)
            continue;

        // Perform collapse
        vertex_collapse_map[polymesh.property(v_id_tag, vh_from)] = polymesh.property(v_id_tag, vh_to);
        if (should_swap && !swaped)
        {
            if (vertex_feature_tag[corresponding_mesh_vertex_ids[vh_from_idx]] && !vertex_feature_tag[corresponding_mesh_vertex_ids[vh_to_idx]])
            {
                corresponding_mesh_vertex_ids[vh_to_idx] = corresponding_mesh_vertex_ids[vh_from_idx];
            }
            if (vh_from_feature_type != -1 && vh_to_feature_type == -1)
            {
                polymesh.property(v_feature_tag, vh_to) = vh_from_feature_type;
            }
        }

        polymesh.collapse(heh);

        // add all edges of the faces adjacent to vh_to to the queue
        std::unordered_set<PolyMesh::EdgeHandle> edge_set;
        for (auto vf_it = polymesh.cvf_iter(vh_to); vf_it.is_valid(); ++vf_it)
        {
            auto fh = *vf_it;
            for (auto fe_it = polymesh.cfe_iter(fh); fe_it.is_valid(); ++fe_it)
            {
                auto heh_adj = polymesh.halfedge_handle(*fe_it, 0);
                auto edge = polymesh.edge_handle(heh_adj);
                if (edge_set.find(edge) != edge_set.end())
                    continue;
                edge_set.insert(edge);
                EdgeCollapseInfo ec(polymesh, edge);
                if (ec.edge_type <= edge_type_threshold)
                    edge_queue.push(ec);
            }
        }
    }
    // Garbage collection
    polymesh.garbage_collection();
    std::vector<ptrdiff_t> vertex_remap(quad_vertices.size(), -1);
    for (size_t i = 0; i < quad_vertices.size(); i++)
    {
        ptrdiff_t current_id = (ptrdiff_t)i;
        while (vertex_collapse_map.find(current_id) != vertex_collapse_map.end())
        {
            current_id = vertex_collapse_map[current_id];
        }
        vertex_remap[i] = current_id;
    }

    std::unordered_map<ptrdiff_t, std::set<ptrdiff_t>> neighbor_cluster_info;
    for (size_t i = 0; i < vertex_remap.size(); i++)
    {
        auto hv = m_pmesh->get_vertex(corresponding_mesh_vertex_ids[i]);
        auto he = hv->edge;
        do
        {
            if (he->face)
            {
                neighbor_cluster_info[vertex_remap[i]].insert(face_cluster_ids[he->face->id]);
            }
            he = he->pair->next;
        } while (he != hv->edge);
    }

    //
    new_quad_vertices.resize(0);
    std::vector<ptrdiff_t> remained_original_vertex_ids;
    remained_original_vertex_ids.reserve(polymesh.n_vertices());
    std::vector<std::set<ptrdiff_t>> vertex_belonging_clusters;
    std::unordered_map<ptrdiff_t, size_t> id_remap;

    for (size_t i = 0; i < vertex_remap.size(); i++)
    {
        if (i == vertex_remap[i])
        {
            remained_original_vertex_ids.push_back(corresponding_mesh_vertex_ids[i]);
            vertex_belonging_clusters.push_back(neighbor_cluster_info[i]);
            new_quad_vertices.emplace_back(quad_vertices[i]);
            id_remap[i] = new_quad_vertices.size() - 1;
        }
    }
    quad_faces.resize(0);
    quad_faces.reserve(polymesh.n_faces());
    int non_quad_num = 0;
    for (const auto &f : polymesh.faces())
    {
        std::vector<size_t> face;
        for (auto fv_it = polymesh.cfv_iter(f); fv_it.is_valid(); ++fv_it)
        {
            face.push_back(id_remap[polymesh.property(v_id_tag, *fv_it)]);
        }
        quad_faces.push_back(face);
        if (face.size() != 4)
            non_quad_num++;
    }

    quad_vertices = new_quad_vertices;
    //

    bool applied_fixing = false;
    if (non_quad_num > 0)
    {
        if (m_verbose)
        {
            std::cout << "Warning: " << non_quad_num << " non-quad polygons exist in the extracted quad mesh!" << std::endl;
        }

        if (improve_mode)
        {
            non_quad_num = fix_nonquad_faces(non_quad_num);
            applied_fixing = true;
            if (m_verbose)
            {
                if (non_quad_num > 0)
                    std::cout << "Warning: " << non_quad_num << " non-quad polygons still exist in the extracted quad mesh after fixing!" << std::endl;
                else
                    std::cout << "All non-quad polygons are successfully fixed!" << std::endl;
            }
        }
    }

    // apply quad split subdivision and smoothing
    if (subdiv_num == 0 || applied_fixing)
        return;

    ///////////////////////////////////////////////////////////////////////
    std::vector<ig::AABB_Segment_Tree<Real> *> feature_arc_trees(feature_edge_loops.size(), 0);
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)feature_edge_loops.size(); i++)
    {
        std::vector<TinyVector<Real, 3>> arc_points;
        for (size_t j = 0; j < feature_edge_loops[i].size() - 1; j++)
        {
            arc_points.emplace_back(feature_edge_loops[i][j]->pos);
            arc_points.emplace_back(feature_edge_loops[i][j + 1]->pos);
        }
        feature_arc_trees[i] = new ig::AABB_Segment_Tree<Real>(arc_points);
    }
    auto cluster_size = *std::max_element(face_cluster_ids.begin(), face_cluster_ids.end()) + 1;
    std::vector<std::vector<TinyVector<Real, 3>>> cluster_faces(cluster_size);
    std::vector<ig::AABB_Tree<Real> *> cluster_trees(cluster_size, 0);
    for (ptrdiff_t i = 0; i < (ptrdiff_t)m_pmesh->get_num_of_faces(); i++)
    {
        auto cid = face_cluster_ids[i];
        auto he = m_pmesh->get_face(i)->edge;
        cluster_faces[cid].emplace_back(he->pair->vert->pos);
        cluster_faces[cid].emplace_back(he->vert->pos);
        cluster_faces[cid].emplace_back(he->next->vert->pos);
    }
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)cluster_faces.size(); i++)
    {
        if (!cluster_faces[i].empty())
            cluster_trees[i] = new ig::AABB_Tree<Real>(cluster_faces[i]);
    }
    std::unordered_map<ptrdiff_t, std::set<int>> tri_vertex_to_feature_arc_ids;
    for (size_t i = 0; i < feature_edge_loops.size(); i++)
    {
        for (size_t j = 0; j < feature_edge_loops[i].size(); j++)
        {
            tri_vertex_to_feature_arc_ids[feature_edge_loops[i][j]->id].insert((int)i);
        }
    }
    std::unordered_set<ptrdiff_t> feature_vertices;
    std::unordered_map<ptrdiff_t, std::set<int>> vertex_to_feature_arc_ids;
    // std::vector<std::set<ptrdiff_t>> vertex_belonging_clusters(quad_vertices.size());
    for (ptrdiff_t i = 0; i < (ptrdiff_t)remained_original_vertex_ids.size(); i++)
    {
        auto vid = remained_original_vertex_ids[i];
        if (vertex_feature_tag[vid])
            feature_vertices.insert(i);
        auto iter = tri_vertex_to_feature_arc_ids.find(vid);
        if (iter != tri_vertex_to_feature_arc_ids.end())
            vertex_to_feature_arc_ids[i] = iter->second;
        // auto hv = m_pmesh->get_vertex(vid);
        // auto he = hv->edge;
        // do
        // {
        //     if (he->face)
        //     {
        //         auto cid = face_cluster_ids[he->face->id];
        //         vertex_belonging_clusters[i].insert(cid);
        //     }
        //     he = he->pair->next;
        // } while (he != hv->edge);
    }

    auto mesh = create_mesh(quad_vertices, quad_faces, true);

    for (ptrdiff_t i = 0; i < (ptrdiff_t)mesh->get_num_of_vertices(); i++)
    {
        auto hv = mesh->get_vertex(i);
        bool singular = mesh->is_on_boundary(hv) ? hv->degree != 3 : hv->degree != 4;
        if (singular)
        {
            feature_vertices.insert(i);
        }
    }

    for (unsigned int iter = 0; iter < subdiv_num; iter++)
    {
        MeshLib::MeshSubdivision<Real> sub(mesh);
        auto submesh = sub.SplitQuad4CDF(vertex_belonging_clusters, vertex_to_feature_arc_ids);
        std::swap(mesh, submesh);
        delete submesh;
        quad_vertices.resize(mesh->get_num_of_vertices());
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < (ptrdiff_t)mesh->get_num_of_vertices(); i++)
        {
            quad_vertices[i] = mesh->get_vertex(i)->pos;
        }
        quad_smoothing(mesh, vertex_belonging_clusters, vertex_to_feature_arc_ids, feature_vertices, cluster_trees, feature_arc_trees);
    }
    // quad_smoothing(mesh, vertex_belonging_clusters, vertex_to_feature_arc_ids, feature_vertices, cluster_trees, feature_arc_trees);

    quad_faces.resize(mesh->get_num_of_faces());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)mesh->get_num_of_faces(); i++)
    {
        auto he = mesh->get_face(i)->edge;
        quad_faces[i].resize(0);
        do
        {
            quad_faces[i].push_back(he->pair->vert->id);
            he = he->next;
        } while (he != mesh->get_face(i)->edge);
    }
    delete mesh;

    // destroy the trees
    for (size_t i = 0; i < feature_arc_trees.size(); i++)
    {
        if (feature_arc_trees[i])
            delete feature_arc_trees[i];
    }
    for (size_t i = 0; i < cluster_trees.size(); i++)
    {
        if (cluster_trees[i])
            delete cluster_trees[i];
    }

    // std::vector<TinyVector<Real, 3>> fixed_vertices;
    // for (size_t i = 0; i < quad_vertices.size(); i++)
    // {
    //     if (feature_vertices.find(i) != feature_vertices.end())
    //     fixed_vertices.emplace_back(quad_vertices[i]);
    // }
    // SavePtsPLY(path_concatenate(debug_dir, "fixed.ply"), fixed_vertices);
}
////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::merge_clusters(const FaceClusterType type)
{
    // return;
    // const Real color_bound = (Real)0.5;
    // std::unordered_set<MySortedTuple<ptrdiff_t, 2, true>> cluster_edges;
    const std::vector<Real> &face_colors = (type == FaceClusterType::FC_DCDF) ? face_dcdf_colors : face_cdf_colors;
    // for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i += 2)
    // {
    //     auto edge = m_pmesh->get_edge(i);
    //     if (edge->face && edge->pair->face)
    //     {
    //         const auto &cid0 = face_cluster_ids[edge->face->id];
    //         const auto &cid1 = face_cluster_ids[edge->pair->face->id];
    //         if (cid0 != cid1 && cid0 > 0 && cid1 > 0 &&
    //             (face_colors[edge->face->id] > color_bound || face_colors[edge->pair->face->id] > color_bound))
    //         {
    //             MySortedTuple<ptrdiff_t, 2, true> edge_key(cid0, cid1);
    //             cluster_edges.insert(edge_key);
    //         }
    //     }
    // }
    std::unordered_map<MySortedTuple<ptrdiff_t, 2, true>, std::pair<Real, Real>> adjacency_cluster_info;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i += 2)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge->face && edge->pair->face)
        {
            const auto &cid0 = face_cluster_ids[edge->face->id];
            const auto &cid1 = face_cluster_ids[edge->pair->face->id];
            if (cid0 != cid1 && cid0 > 0 && cid1 > 0)
            {
                Real edge_length = edge->GetLength();
                MySortedTuple<ptrdiff_t, 2, true> edge_key(cid0, cid1);
                bool status = face_colors[edge->face->id] > mid_color || face_colors[edge->pair->face->id] > mid_color;
                Real confuse_length = status ? edge_length : 0;
                Real clear_length = status ? 0 : edge_length;

                auto iter = adjacency_cluster_info.find(edge_key);
                if (iter == adjacency_cluster_info.end())
                {
                    adjacency_cluster_info[edge_key] = std::make_pair(confuse_length, clear_length);
                }
                else
                {
                    iter->second.first += confuse_length;
                    iter->second.second += clear_length;
                }
            }
        }
    }
    std::vector<MySortedTuple<ptrdiff_t, 2, true>> cluster_edges;
    for (const auto &item : adjacency_cluster_info)
    {
        Real confuse_length = item.second.first;
        Real clear_length = item.second.second;
        if (confuse_length > clear_length)
        {
            cluster_edges.push_back(item.first);
        }
    }

    ////////////////////////////////////////////////////////

    std::unordered_map<ptrdiff_t, std::unordered_set<ptrdiff_t>> cluster_adjacency;
    for (const auto &edge : cluster_edges)
    {
        cluster_adjacency[edge[0]].insert(edge[1]);
        cluster_adjacency[edge[1]].insert(edge[0]);
    }
    std::vector<ptrdiff_t> cluster_ids;
    for (const auto &item : cluster_adjacency)
    {
        cluster_ids.push_back(item.first);
    }
    for (auto cid : cluster_ids)
    {
        auto &adjacent_clusters = cluster_adjacency[cid];
        if (adjacent_clusters.empty())
            continue;
        bool processed = false;
        do
        {
            processed = false;
            std::unordered_set<ptrdiff_t> merged_clusters = adjacent_clusters;
            for (const auto &adj_cid : adjacent_clusters)
            {
                if (cluster_adjacency[adj_cid].empty())
                    continue;
                merged_clusters.insert(cluster_adjacency[adj_cid].begin(), cluster_adjacency[adj_cid].end());
                cluster_adjacency[adj_cid].clear();
            }
            merged_clusters.erase(cid);
            if (merged_clusters.size() > adjacent_clusters.size())
            {
                adjacent_clusters = merged_clusters;
                processed = true;
            }
        } while (processed);
    }

    std::unordered_map<ptrdiff_t, ptrdiff_t> cluster_reindex_map;
    for (const auto &item : cluster_adjacency)
    {
        ptrdiff_t cid = item.first;
        for (const auto &adj_cid : item.second)
        {
            cluster_reindex_map[adj_cid] = cid;
        }
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        if (face_cluster_ids[i] > 0)
        {
            auto iter = cluster_reindex_map.find(face_cluster_ids[i]);
            if (iter != cluster_reindex_map.end())
                face_cluster_ids[i] = iter->second;
        }
    }
}
////////////////////////////////
template <typename Real>
int QCDF2QuadMesh<Real>::fix_nonquad_faces(const int non_quad_num)
{

    improve_mesh_topology(quad_vertices, quad_faces);
    int non_quad_count = 0;
    for (size_t i = 0; i < quad_faces.size(); i++)
    {
        if (quad_faces[i].size() != 4)
            non_quad_count++;
    }
    return non_quad_count;
    std::unordered_map<MySortedTuple<ptrdiff_t, 2, true>, MySortedTuple<ptrdiff_t, 2, false>> edge_to_face_map;

    for (size_t i = 0; i < quad_faces.size(); i++)
    {
        for (size_t j = 0; j < quad_faces[i].size(); j++)
        {
            ptrdiff_t v0 = quad_faces[i][j];
            ptrdiff_t v1 = quad_faces[i][(j + 1) % quad_faces[i].size()];
            MySortedTuple<ptrdiff_t, 2, true> edge_key(v0, v1);
            auto iter = edge_to_face_map.find(edge_key);
            if (iter != edge_to_face_map.end())
            {
                if (iter->second[0] == -1)
                    iter->second[0] = i;
                else
                    iter->second[1] = i;
            }
            else
            {
                if (v0 < v1)
                    edge_to_face_map[edge_key] = MySortedTuple<ptrdiff_t, 2, false>(i, -1);
                else
                    edge_to_face_map[edge_key] = MySortedTuple<ptrdiff_t, 2, false>(-1, i);
            }
        }
    }
    std::vector<int> vertex_degrees(quad_vertices.size(), 0);
    for (const auto &item : edge_to_face_map)
    {
        vertex_degrees[item.first[0]]++;
        vertex_degrees[item.first[1]]++;
    }
    std::vector<bool> valid_faces(quad_faces.size(), true);
    std::vector<std::vector<size_t>> new_quad_faces = quad_faces;
    std::vector<size_t> new_quad(4);

    bool processed = false;
    do
    {
        processed = false;

        for (size_t i = 0; i < new_quad_faces.size(); i++)
        {
            if (new_quad_faces[i].size() != 4 || valid_faces[i] == false)
                continue;
            for (size_t j = 0; j < quad_faces[i].size(); j++)
            {
                ptrdiff_t v0 = new_quad_faces[i][j];
                ptrdiff_t v1 = new_quad_faces[i][(j + 1) % new_quad_faces[i].size()];
                ptrdiff_t v2 = new_quad_faces[i][(j + 2) % new_quad_faces[i].size()];
                ptrdiff_t v3 = new_quad_faces[i][(j + 3) % new_quad_faces[i].size()];
                MySortedTuple<ptrdiff_t, 2, true> key_01(v0, v1), key_12(v1, v2), key_23(v2, v3), key_30(v3, v0);
                auto edge_01 = edge_to_face_map.find(key_01);
                auto pair_faces_01 = v0 < v1 ? edge_01->second[1] : edge_01->second[0];
                auto edge_12 = edge_to_face_map.find(key_12);
                auto pair_faces_12 = v1 < v2 ? edge_12->second[1] : edge_12->second[0];
                auto edge_23 = edge_to_face_map.find(key_23);
                auto pair_faces_23 = v2 < v3 ? edge_23->second[1] : edge_23->second[0];
                auto edge_30 = edge_to_face_map.find(key_30);

                if (pair_faces_01 >= 0 && pair_faces_12 >= 0 &&
                    new_quad_faces[pair_faces_01].size() == 3 && new_quad_faces[pair_faces_12].size() == 3 &&
                    valid_faces[pair_faces_01] && valid_faces[pair_faces_12])
                {
                    ptrdiff_t opp_v_01 = -1, opp_v_12 = -1;
                    for (int k = 0; k < 3; k++)
                    {
                        if (new_quad_faces[pair_faces_01][k] != v0 && new_quad_faces[pair_faces_01][k] != v1)
                        {
                            opp_v_01 = new_quad_faces[pair_faces_01][k];
                        }
                        if (new_quad_faces[pair_faces_12][k] != v1 && new_quad_faces[pair_faces_12][k] != v2)
                        {
                            opp_v_12 = new_quad_faces[pair_faces_12][k];
                        }
                    }
                    MySortedTuple<ptrdiff_t, 2, true> e13(v1, v3);
                    if (opp_v_01 != v3 && opp_v_12 != v0 && edge_to_face_map.find(e13) == edge_to_face_map.end())
                    {
                        valid_faces[pair_faces_12] = false;
                        ptrdiff_t new_face_id_0 = i;
                        ptrdiff_t new_face_id_1 = pair_faces_01;

                        new_quad[0] = v3, new_quad[1] = v0, new_quad[2] = opp_v_01, new_quad[3] = v1;
                        new_quad_faces[new_face_id_0] = new_quad;
                        new_quad[0] = v2, new_quad[1] = v3, new_quad[2] = v1, new_quad[3] = opp_v_12;
                        new_quad_faces[new_face_id_1] = new_quad;
                        vertex_degrees[v1]--, vertex_degrees[v3]++;
                        vertex_degrees[v0]--, vertex_degrees[v2]--;

                        edge_to_face_map.erase(edge_01);
                        edge_to_face_map.erase(edge_12);

                        MySortedTuple<ptrdiff_t, 2, true> e13(v1, v3);
                        if (v1 < v3)
                            edge_to_face_map[e13] = MySortedTuple<ptrdiff_t, 2, false>(new_face_id_0, new_face_id_1);
                        else
                            edge_to_face_map[e13] = MySortedTuple<ptrdiff_t, 2, false>(new_face_id_1, new_face_id_0);

                        // update 03, 23, 0-01, 1-01, 1-12, 2-12
                        auto edge_30 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v0, v3));
                        auto edge_0_01 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v0, opp_v_01));
                        auto edge_1_01 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v1, opp_v_01));
                        auto edge_1_12 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v1, opp_v_12));
                        auto edge_2_12 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v2, opp_v_12));

                        edge_30->second[v3 > v0] = new_face_id_0;
                        edge_0_01->second[v0 > opp_v_01] = new_face_id_0;
                        edge_1_01->second[opp_v_01 > v1] = new_face_id_0;
                        edge_1_12->second[v1 > opp_v_12] = new_face_id_1;
                        edge_2_12->second[opp_v_12 > v2] = new_face_id_1;
                        edge_23->second[v2 > v3] = new_face_id_1;

                        processed = true;
                        break;
                    }
                }

                if (pair_faces_01 >= 0 && pair_faces_23 >= 0 &&
                    new_quad_faces[pair_faces_01].size() == 3 && new_quad_faces[pair_faces_23].size() == 3)
                {
                    ptrdiff_t opp_v_01 = -1, opp_v_23 = -1;
                    for (int k = 0; k < 3; k++)
                    {
                        if (quad_faces[pair_faces_01][k] != v0 && quad_faces[pair_faces_01][k] != v1)
                        {
                            opp_v_01 = quad_faces[pair_faces_01][k];
                        }
                        if (quad_faces[pair_faces_23][k] != v2 && quad_faces[pair_faces_23][k] != v3)
                        {
                            opp_v_23 = quad_faces[pair_faces_23][k];
                        }
                    }
                    if (std::max(vertex_degrees[v0], vertex_degrees[v2]) > std::max(vertex_degrees[v1], vertex_degrees[v3]))
                    {
                        MySortedTuple<ptrdiff_t, 2, true> e13(v1, v3);
                        if (opp_v_01 != v3 && opp_v_23 != v1 && edge_to_face_map.find(e13) == edge_to_face_map.end())
                        {
                            ptrdiff_t new_face_id_0 = i;
                            ptrdiff_t new_face_id_1 = pair_faces_01;
                            valid_faces[pair_faces_23] = false;

                            new_quad[0] = v0, new_quad[1] = opp_v_01, new_quad[2] = v1, new_quad[3] = v3;
                            new_quad_faces[new_face_id_0] = new_quad;
                            new_quad[0] = v3, new_quad[1] = v1, new_quad[2] = v2, new_quad[3] = opp_v_23;
                            new_quad_faces[new_face_id_1] = new_quad;
                            vertex_degrees[v0]--, vertex_degrees[v2]--;

                            edge_to_face_map.erase(edge_01);
                            edge_to_face_map.erase(edge_23);

                            if (v1 < v3)
                                edge_to_face_map[e13] = MySortedTuple<ptrdiff_t, 2, false>(new_face_id_0, new_face_id_1);
                            else
                                edge_to_face_map[e13] = MySortedTuple<ptrdiff_t, 2, false>(new_face_id_1, new_face_id_0);

                            // 0-01, 1-01, 03, 12, 2-23, 23-3
                            auto edge_0_01 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v0, opp_v_01));
                            auto edge_1_01 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v1, opp_v_01));
                            auto edge_2_23 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v2, opp_v_23));
                            auto edge_23_3 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v3, opp_v_23));

                            edge_0_01->second[v0 > opp_v_01] = new_face_id_0;
                            edge_1_01->second[opp_v_01 > v1] = new_face_id_0;
                            edge_30->second[v3 > v0] = new_face_id_0;
                            edge_12->second[v1 > v2] = new_face_id_1;
                            edge_2_23->second[v2 > opp_v_23] = new_face_id_1;
                            edge_23_3->second[opp_v_23 > v3] = new_face_id_1;
                            processed = true;
                            break;
                        }
                    }
                    else
                    {
                        MySortedTuple<ptrdiff_t, 2, true> e02(v0, v2);
                        if (opp_v_01 != v2 && opp_v_23 != v0 && edge_to_face_map.find(e02) == edge_to_face_map.end())
                        {
                            ptrdiff_t new_face_id_0 = i;
                            ptrdiff_t new_face_id_1 = pair_faces_01;
                            valid_faces[pair_faces_23] = false;

                            new_quad[0] = v0, new_quad[1] = opp_v_01, new_quad[2] = v1, new_quad[3] = v2;
                            new_quad_faces[new_face_id_0] = new_quad;
                            new_quad[0] = v3, new_quad[1] = v0, new_quad[2] = v2, new_quad[3] = opp_v_23;
                            new_quad_faces[new_face_id_1] = new_quad;
                            vertex_degrees[v1]--, vertex_degrees[v3]--;

                            edge_to_face_map.erase(edge_01);
                            edge_to_face_map.erase(edge_23);

                            if (v2 < v0)
                                edge_to_face_map[e02] = MySortedTuple<ptrdiff_t, 2, false>(new_face_id_0, new_face_id_1);
                            else
                                edge_to_face_map[e02] = MySortedTuple<ptrdiff_t, 2, false>(new_face_id_1, new_face_id_0);

                            // 0-01, 1-01, 03, 12, 2-23, 23-3
                            auto edge_0_01 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v0, opp_v_01));
                            auto edge_1_01 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v1, opp_v_01));
                            auto edge_2_23 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v2, opp_v_23));
                            auto edge_23_3 = edge_to_face_map.find(MySortedTuple<ptrdiff_t, 2, true>(v3, opp_v_23));

                            edge_0_01->second[v0 > opp_v_01] = new_face_id_0;
                            edge_1_01->second[opp_v_01 > v1] = new_face_id_0;
                            edge_12->second[v1 > v2] = new_face_id_0;
                            edge_30->second[v3 > v0] = new_face_id_1;
                            edge_2_23->second[v2 > opp_v_23] = new_face_id_1;
                            edge_23_3->second[opp_v_23 > v3] = new_face_id_1;
                            processed = true;
                            break;
                        }
                    }
                }
            }
        }
    } while (processed);
    int remained_nonquad = 0;
    quad_faces.resize(0);
    for (size_t i = 0; i < new_quad_faces.size(); i++)
    {
        if (valid_faces[i])
        {
            quad_faces.emplace_back(new_quad_faces[i]);
            if (new_quad_faces[i].size() != 4)
                remained_nonquad++;
        }
    }
    return remained_nonquad;
}
////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::quad_extraction(const FaceClusterType method)
{
    auto nf = m_pmesh->get_num_of_faces();

    int cdf_patch_num = 0, dcdf_patch_num = 0;
    std::vector<FaceClusterType> type_store;
    if (method == FaceClusterType::FC_CDF_DCDF)
    {
        type_store.push_back(FaceClusterType::FC_DCDF);
        type_store.push_back(FaceClusterType::FC_CDF);
    }
    else
    {
        type_store.push_back(method);
    }

    // init();

    std::vector<std::vector<ptrdiff_t>> face_cluster_ids_store;

    for (const auto &type : type_store)
    {
        std::string type_str = (type == FaceClusterType::FC_CDF) ? "cdf_" : "dcdf_";

        // Step 1: face color polarization, and use the polarized color for initial clustering, isolated regions by the black bands
        int num_clusters = cluster_via_polarization(type);
        if (m_debug_mode)
        {
            std::cout << (type == FaceClusterType::FC_CDF ? "CDF" : "DCDF") << " processing" << std::endl;
            std::cout << "Step 1 (color polarization) is done!" << std::endl;
            SavePLYmesh_with_float_storage(m_pmesh, path_concatenate(debug_dir, type_str + "clustered_face.ply"), &face_cluster_ids, false);

            const std::vector<Real> &face_colors = (type == FaceClusterType::FC_CDF) ? face_cdf_colors : face_dcdf_colors;
            SavePLYmesh_with_float_storage_and_gray_color(m_pmesh, path_concatenate(debug_dir, type_str + "color_faces.ply"), &face_colors, true, true);

            if (!vertex_cdf_colors.empty())
            {
                const std::vector<Real> &vertex_colors = (type == FaceClusterType::FC_CDF) ? vertex_cdf_colors : vertex_dcdf_colors;
                SavePLYmesh_with_float_storage_and_gray_color(m_pmesh, path_concatenate(debug_dir, type_str + "color_vertices.ply"), &vertex_colors, false, true);
            }
        }

        // Step 2: process unclustered faces, assign them to the closest cluster using both cdf and dcdf directions
        process_uncluster_faces(type, num_clusters);
        if (m_debug_mode)
        {
            std::cout << "Step 2 (process unclustered faces) is done!" << std::endl;
            SavePLYmesh_with_float_storage(m_pmesh, path_concatenate(debug_dir, type_str + "clustered_face2.ply"), &face_cluster_ids, false);
        }

        // verify whether all faces are clustered
        ptrdiff_t unclustered_face_count = 0;
#pragma omp parallel for reduction(+ : unclustered_face_count)
        for (ptrdiff_t i = 0; i < nf; i++)
        {
            if (face_cluster_ids[i] <= 0)
                unclustered_face_count++;
        }
        if (unclustered_face_count > 0)
        {
            if (m_verbose)
            {
                std::cout << "Warning: " << unclustered_face_count << " unclustered faces remain after Step 2!" << std::endl;
            }
        }

        // Step 3: remove interior clusters which have no more than two neighboring clusters; and reassign the faces to the neighboring clusters
        // bool handled = false;
        // bool has_incorrect_clusters = false;
        // do
        // {
        //     handled = handle_incorrect_clusters(type, num_clusters);
        //     if (handled)
        //         has_incorrect_clusters = true;
        // } while (handled);

        int num_reindexed = reindex_clusters(); // re-index the cluster ids from 0 to num_clusters-1.
        if (type == FaceClusterType::FC_CDF)
            cdf_patch_num = num_reindexed;
        else
            dcdf_patch_num = num_reindexed;
        if (m_debug_mode)
        {
            // if (handled)
            //     std::cout << "Step 3 (handle incorrect clusters) is done!" << std::endl;
            // else
            //     std::cout << "Step 3 (handle incorrect clusters): no incorrect cluster found!" << std::endl;
            SavePLYmesh_with_float_storage(m_pmesh, path_concatenate(debug_dir, type_str + "final_clustered_face.ply"), &face_cluster_ids, false);

            SaveChartEdge_as_ply(m_pmesh, path_concatenate(debug_dir, type_str + "final_clustered_edge.ply"), face_cluster_ids, 0.001f * (float)global_scale);
        }

        if (method == FaceClusterType::FC_CDF_DCDF)
        {
            face_cluster_ids_store.push_back(face_cluster_ids);
        }
    }

    if (method == FaceClusterType::FC_CDF_DCDF)
    {
        update_face_cluster_ids(face_cluster_ids_store, cdf_patch_num);
        if (m_debug_mode)
        {
            SavePLYmesh_with_float_storage(m_pmesh, path_concatenate(debug_dir, "cdf_dcdf_final_clustered_face.ply"), &face_cluster_ids, false);

            SaveChartEdge_as_ply(m_pmesh, path_concatenate(debug_dir, "cdf_dcdf_final_clustered_edge.ply"), face_cluster_ids, 0.001f * (float)global_scale);
        }
    }

    // Step 4: generate the quad mesh using the face clusters
    if (method == FaceClusterType::FC_DCDF)
        dcdf_build_quad_patches();
    else
        cdf_build_quad_patches();

    if (m_debug_mode)
    {
        std::vector<TinyVector<Real, 3>> debug_vertices;
        std::vector<std::vector<size_t>> debug_faces;
        for (size_t i = 0; i < quad_faces.size(); i++)
        {
            if (quad_faces[i].size() != 4)
            {
                std::vector<size_t> face;
                for (size_t j = 0; j < quad_faces[i].size(); j++)
                {
                    debug_vertices.push_back(quad_vertices[quad_faces[i][j]]);
                    face.push_back(debug_vertices.size() - 1);
                }
                debug_faces.push_back(face);
            }
        }
        if (!debug_faces.empty())
        {
            std::vector<size_t> debug_cluster_ids(debug_faces.size(), 0);
            SavePLYMesh_with_color(path_concatenate(debug_dir, "non_quad_polygons.ply"), debug_vertices, debug_faces, 0, &debug_cluster_ids);
        }
    }
}
////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::update_face_cluster_ids(const std::vector<std::vector<ptrdiff_t>> &face_cluster_ids_store, const int cdf_patch_num)
{
    ptrdiff_t nf = m_pmesh->get_num_of_faces();
    const auto &cdf_face_cluster_ids = face_cluster_ids_store[0];
    const auto &dcdf_face_cluster_ids = face_cluster_ids_store[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i += 2)
    {
        auto edge = m_pmesh->get_edge(i);
        auto pair_edge = edge->pair;
        if (!edge->face || !pair_edge->face || cdf_face_cluster_ids[edge->face->id] != cdf_face_cluster_ids[pair_edge->face->id] ||
            dcdf_face_cluster_ids[pair_edge->face->id] != dcdf_face_cluster_ids[edge->face->id])
        {
            edge->tag = pair_edge->tag = true;
        }
        else
        {
            edge->tag = pair_edge->tag = false;
        }
    }

    ptrdiff_t cluster_id = 0;
    face_cluster_ids.assign(nf, -1);

    for (ptrdiff_t i = 0; i < nf; i++)
    {
        if (face_cluster_ids[i] == -1)
        {
            // start a new cluster
            std::queue<ptrdiff_t> face_queue;
            face_queue.push(i);
            face_cluster_ids[i] = cluster_id;

            while (!face_queue.empty())
            {
                ptrdiff_t fid = face_queue.front();
                face_queue.pop();
                auto face = m_pmesh->get_face(fid);
                auto edge = face->edge;
                do
                {
                    auto pair_edge = edge->pair;
                    if (pair_edge->face && !edge->tag)
                    {
                        ptrdiff_t adj_fid = pair_edge->face->id;
                        if (face_cluster_ids[adj_fid] == -1)
                        {
                            face_cluster_ids[adj_fid] = cluster_id;
                            face_queue.push(adj_fid);
                        }
                    }
                    edge = edge->next;
                } while (edge != face->edge);
            }
            cluster_id++;
        }
    }

    // update tri_mesh_vertex_cdfdcdf_types
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto hv = m_pmesh->get_vertex(i);
        auto he = hv->edge;
        std::unordered_set<ptrdiff_t> cdfcluster_id_set, dcdfcluster_id_set;
        do
        {
            if (he->face)
            {
                ptrdiff_t fid = he->face->id;
                cdfcluster_id_set.insert(cdf_face_cluster_ids[fid]);
                dcdfcluster_id_set.insert(dcdf_face_cluster_ids[fid]);
            }
            he = he->pair->next;
        } while (he != hv->edge);
        if (cdfcluster_id_set.size() == 1 && dcdfcluster_id_set.size() > 1)
            tri_mesh_vertex_cdfdcdf_types[i] = 1;
        else if (dcdfcluster_id_set.size() == 1 && cdfcluster_id_set.size() > 1)
            tri_mesh_vertex_cdfdcdf_types[i] = 2;
        else if (cdfcluster_id_set.size() > 1 && dcdfcluster_id_set.size() > 1)
            tri_mesh_vertex_cdfdcdf_types[i] = 0;
    }

    // std::vector<TinyVector<Real, 3>> tmp_points;
    // std::vector<size_t> vertex_cluster_ids;
    // for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    // {
    //     if (tri_mesh_vertex_cdfdcdf_types[i] != -1)
    //     {
    //         tmp_points.push_back(m_pmesh->get_vertex(i)->pos);
    //         vertex_cluster_ids.push_back(tri_mesh_vertex_cdfdcdf_types[i]);
    //     }
    // }
    // SavePLYMesh_with_color(path_concatenate(debug_dir, "cdf_dcdf_type_vertices.ply"), tmp_points, std::vector<std::vector<size_t>>(), &vertex_cluster_ids, nullptr);

    // eliminate small clusters: as CDF patches are known.
    std::vector<Real> cluster_areas(cluster_id, (Real)0);
    std::vector<ptrdiff_t> orginal_cluster_ids(cluster_id, 0);
    for (int i = 0; i < cluster_id; i++)
    {
        orginal_cluster_ids[i] = i;
    }
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nf; i++)
    {
        cluster_areas[face_cluster_ids[i]] += face_areas[i];
    }

    if (4 * cdf_patch_num > cluster_id)
    {
        if (m_verbose)
            std::cerr << "Warning: the number of clusters is less than or equal to 4 times of CDF patches!" << std::endl;
        return;
    }
    else if (4 * cdf_patch_num == cluster_id)
    {
        return;
    }
    // sort the clusters based on their areas
    std::sort(orginal_cluster_ids.begin(), orginal_cluster_ids.end(),
              [&cluster_areas](const ptrdiff_t &a, const ptrdiff_t &b)
              { return cluster_areas[a] > cluster_areas[b]; });
    std::vector<ptrdiff_t> cluster_reindex_map(cluster_id, -1);
    for (int i = 0; i < 4 * cdf_patch_num; i++)
    {
        cluster_reindex_map[orginal_cluster_ids[i]] = i;
    }
    std::unordered_map<ptrdiff_t, std::unordered_set<ptrdiff_t>> cluster_pair_map;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i += 2)
    {
        auto edge = m_pmesh->get_edge(i);
        auto pair_edge = edge->pair;
        if (edge->face && pair_edge->face)
        {
            ptrdiff_t cid0 = face_cluster_ids[edge->face->id];
            ptrdiff_t cid1 = face_cluster_ids[pair_edge->face->id];
            if (cid0 != cid1)
            {
                cluster_pair_map[cid0].insert(cid1);
                cluster_pair_map[cid1].insert(cid0);
            }
        }
    }

    for (int i = 4 * cdf_patch_num; i < cluster_id; i++)
    {
        ptrdiff_t map_id = -1;
        Real max_area = 0;
        for (const auto &adj_cid : cluster_pair_map[orginal_cluster_ids[i]])
        {
            if (cluster_areas[adj_cid] > max_area && cluster_reindex_map[adj_cid] >= 0)
            {
                max_area = cluster_areas[adj_cid];
                map_id = cluster_reindex_map[adj_cid];
            }
        }
        if (map_id >= 0)
            cluster_reindex_map[orginal_cluster_ids[i]] = map_id;
        else if (m_verbose)
            std::cout << "Warning: cannot find a valid cluster to map for cluster " << orginal_cluster_ids[i] << std::endl;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nf; i++)
    {
        face_cluster_ids[i] = cluster_reindex_map[face_cluster_ids[i]];
    }
}
//////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::quad_smoothing(
    MeshLib::Mesh3D<Real> *mesh,
    const std::vector<std::set<ptrdiff_t>> &vertex_belonging_clusters,
    const std::unordered_map<ptrdiff_t, std::set<int>> &vertex_to_feature_arc_ids,
    const std::unordered_set<ptrdiff_t> &feature_vertices,
    const std::vector<ig::AABB_Tree<Real> *> &cluster_trees,
    const std::vector<ig::AABB_Segment_Tree<Real> *> &feature_arc_trees)
{

    std::vector<TinyVector<Real, 3>> new_positions = quad_vertices;
    std::vector<int> nonsmooth_tags(mesh->get_num_of_vertices(), 0);
    Real angle_degree_bound = 1;
    std::vector<std::vector<ptrdiff_t>> vertex_one_rings(mesh->get_num_of_vertices());

#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)mesh->get_num_of_vertices(); i++)
    {
        auto hv = mesh->get_vertex(i);
        if (feature_vertices.find(i) != feature_vertices.end())
        {
            nonsmooth_tags[i] = 1;
            continue;
        }
        // continue;

        bool is_on_boundary = mesh->is_on_boundary(hv);
        // if (is_on_boundary)
        //     nonsmooth_tags[i] = hv->degree != 3 ? 1 : 0;
        // else
        // {
        //     nonsmooth_tags[i] = hv->degree != 4 ? 1 : 0;
        //     if (nonsmooth_tags[i])
        //     {
        //         // if the point is flat, smooth it
        //         auto he = hv->edge;
        //         Real angle_sum = 0;
        //         do
        //         {
        //             angle_sum += compute_angle(he->prev->pair->vert->pos, hv->pos, he->vert->pos);
        //             he = he->pair->next;
        //         } while (he != hv->edge);

        //         if (fabs(angle_sum - 360) < angle_degree_bound)
        //         {
        //             nonsmooth_tags[i] = 0;
        //         }
        //     }
        // }

        vertex_one_rings[i].reserve(hv->degree);
        auto he = hv->edge;
        do
        {
            if (!is_on_boundary)
                vertex_one_rings[i].push_back(he->vert->id);
            else if (mesh->is_on_boundary(he))
            {
                vertex_one_rings[i].push_back(he->vert->id);
            }
            he = he->pair->next;
        } while (he != hv->edge);
    }
    if (smooth_num > 0)
    {
        // std::cout << "Start quad mesh smoothing with " << smooth_num << " iterations." << std::endl;
        for (unsigned int it = 0; it < smooth_num; it++)
        {
            quad_projection(mesh, vertex_belonging_clusters, vertex_to_feature_arc_ids, feature_vertices, cluster_trees, feature_arc_trees, new_positions, nonsmooth_tags);
#pragma omp parallel for schedule(dynamic)
            for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
            {
                if (nonsmooth_tags[i] || vertex_one_rings[i].empty())
                    continue;
                auto hv = mesh->get_vertex(i);
                auto he = hv->edge;
                TinyVector<Real, 3> center(0, 0, 0);
                const auto &one_ring = vertex_one_rings[i];
                Real weight = 0, total_weight = 0;
                if (one_ring.size() % 2 != 0)
                {
                    for (auto j = 0; j < one_ring.size(); j++)
                    {
                        weight = 1;
                        center += quad_vertices[one_ring[j]];
                        total_weight += weight;
                    }
                }
                else
                {
                    for (auto j = 0; j < one_ring.size() / 2; j++)
                    {
                        weight = 1 / ((quad_vertices[one_ring[j]] - quad_vertices[one_ring[(j + one_ring.size() / 2) % one_ring.size()]]).SquaredLength() + (Real)1.0e-12);
                        center += (weight / 2) * (quad_vertices[one_ring[j]] + quad_vertices[one_ring[(j + one_ring.size() / 2) % one_ring.size()]]);
                        total_weight += weight;
                    }
                }
                new_positions[i] = center / total_weight;
            }
        }
    }

    quad_projection(mesh, vertex_belonging_clusters, vertex_to_feature_arc_ids, feature_vertices, cluster_trees, feature_arc_trees, new_positions, nonsmooth_tags);
}
/////////////////////////////////
template <typename Real>
void QCDF2QuadMesh<Real>::quad_projection(
    MeshLib::Mesh3D<Real> *mesh,
    const std::vector<std::set<ptrdiff_t>> &vertex_belonging_clusters,
    const std::unordered_map<ptrdiff_t, std::set<int>> &vertex_to_feature_arc_ids,
    const std::unordered_set<ptrdiff_t> &feature_vertices,
    const std::vector<ig::AABB_Tree<Real> *> &cluster_trees,
    const std::vector<ig::AABB_Segment_Tree<Real> *> &feature_arc_trees,
    const std::vector<TinyVector<Real, 3>> new_positions,
    const std::vector<int> nonsmooth_tags)
{
#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
    {
        auto hv = mesh->get_vertex(i);
        if (nonsmooth_tags[i])
            continue;

        auto iter = vertex_to_feature_arc_ids.find(i);
        if (iter != vertex_to_feature_arc_ids.end())
        {
            auto best_proj = quad_vertices[i];
            TinyVector<Real, 3> proj;
            Real min_dis = std::numeric_limits<Real>::max();
            for (auto eid : iter->second)
            {
                feature_arc_trees[eid]->find_Nearest_Point(new_positions[i], proj);

                Real dis = (proj - new_positions[i]).SquaredLength();
                if (dis < min_dis)
                {
                    min_dis = dis;
                    best_proj = proj;
                }
            }
            quad_vertices[i] = best_proj;
        }
        else if (!vertex_belonging_clusters[i].empty())
        {
            auto best_proj = quad_vertices[i];
            TinyVector<Real, 3> proj;
            Real min_dis = std::numeric_limits<Real>::max();
            for (auto cid : vertex_belonging_clusters[i])
            {
                ptrdiff_t face_ID;
                cluster_trees[cid]->find_Nearest_Point(new_positions[i], proj, &face_ID);
                auto normal = cluster_trees[cid]->get_normal(face_ID);
                // if (hv->normal.Dot(normal) < 0)
                //     continue;
                Real dis = (proj - new_positions[i]).SquaredLength();
                if (dis < min_dis)
                {
                    min_dis = dis;
                    best_proj = proj;
                }
            }

            quad_vertices[i] = best_proj;
        }
        else
        {
            if (m_verbose)
                std::cout << "Warning: vertex " << i << " does not belong to any cluster!" << std::endl;
        }
    }
}
/////////////////////////////////
template class QCDF2QuadMesh<double>;