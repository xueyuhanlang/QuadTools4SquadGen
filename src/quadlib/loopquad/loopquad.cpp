#include "loopquad.h"
#include "loopmerger.h"
#include "blossommerger.h"
#include "quadquality.h"
#include "MeshSubdivision.h"
#include "looputil.h"
#include "MeshLoader.h"
#include <unordered_set>

//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool CheckMeshValidity(MeshLib::Mesh3D<Real> *mesh)
{
    if (!mesh || mesh->get_num_of_faces() < 16)
        return false;

    auto xmax = mesh->xmax - mesh->xmin;
    auto ymax = mesh->ymax - mesh->ymin;
    auto zmax = mesh->zmax - mesh->zmin;
    auto max_range = std::max(std::max(xmax, ymax), zmax);
    auto min_range = std::min(std::min(xmax, ymax), zmax);

    if (min_range / (max_range + 1.0e-12) < (Real)0.001) // too flat
        return false;

    // unsigned int max_degree = 0, min_degree = INT_MAX;
    // for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
    // {
    //     auto vert = mesh->get_vertex(i);
    //     max_degree = std::max(max_degree, vert->degree);
    //     min_degree = std::min(min_degree, vert->degree);
    // }
    // if (max_degree > 8 || min_degree < 3) // too low degree
    //     return false;

    // bool has_singularity = false;
    // for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
    // {
    //     auto hv = mesh->get_vertex(i);
    //     has_singularity = mesh->is_on_boundary(hv) ? hv->degree != 3 : hv->degree != 4;
    // }
    // if (!has_singularity)
    //     return false;

    // if (mesh->is_quad() && (max_degree > 6 || min_degree < 3)) // too high or too low verex degree
    //     return false;

    // if (!mesh->is_quad())
    // {
    //     for (ptrdiff_t i = 0; i < mesh->get_num_of_faces(); i++)
    //     {
    //         auto face = mesh->get_face(i);
    //         if (face->valence != 3)
    //             continue;

    //         auto v0 = face->edge->vert;
    //         auto v1 = face->edge->next->vert;
    //         auto v2 = face->edge->next->next->vert;

    //         if (v0->degree <= 6 || v1->degree <= 6 || v2->degree <= 6) // the triangle is unlikey connected to a high degree vertex, so it is highly possible to be a bad case.
    //             return false;
    //     }
    // }
    return true;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *LoopQuadProcessing(MeshLib::Mesh3D<Real> *input_mesh, const Real loop_threshold, bool debug, bool subdiv, int min_face_num)
{
    if (!input_mesh || input_mesh->get_num_of_faces() < min_face_num)
        return nullptr;

    if (debug)
    {
        std::cout << "Input mesh: " << input_mesh->get_num_of_vertices() << " vertices, "
                  << input_mesh->get_num_of_faces() << " faces, "
                  << input_mesh->get_num_of_edges() << " edges.\n";
    }

    // mesh decomposition
    std::vector<MeshLib::Mesh3D<Real> *> submeshes;
    mesh_decomposition(input_mesh, submeshes);
    if (submeshes.empty())
        return nullptr;
    MeshLib::Mesh3D<Real> *merged_mesh = new MeshLib::Mesh3D<Real>;

    ptrdiff_t num_vertices = 0;

    std::unordered_map<MySortedTuple<ptrdiff_t, 11, false>, std::vector<TinyVector<double, 3>>> database;
    Real bound_threshold = (Real)0.05; //(Real)0.01;
    Real flat_threshold = (Real)0.001;

    for (auto submesh : submeshes)
    {
        //////////////////////////////////////////////////////////////////////

        std::unordered_set<MeshLib::Mesh3D<Real> *> intermediate_meshes;
        intermediate_meshes.insert(submesh);

        MeshLib::Mesh3D<Real> *m_pmesh = nullptr;

        // merge
        if (submesh->is_tri())
        {
            if (subdiv == false)
            {
                if (submesh->get_num_of_faces() % 2 != 0)
                {
                    if (debug)
                    {
                        LoopMerger<Real> merger(submesh);
                        m_pmesh = merger.get_merged_mesh();
                    }
                }
                else
                {
                    try
                    {
                        LoopMerger<Real> merger(submesh);
                        m_pmesh = merger.get_merged_mesh();
                        if (m_pmesh && !m_pmesh->is_quad())
                        {
                            delete m_pmesh;
                            m_pmesh = nullptr;
                            BlossomMerger<Real> merger(submesh);
                            m_pmesh = merger.get_merged_mesh();
                            if (m_pmesh && (m_pmesh->get_num_of_faces() == 0 || (debug == false && !m_pmesh->is_quad())))
                            {
                                delete m_pmesh;
                                m_pmesh = nullptr;
                            }
                        }
                    }
                    catch (...)
                    {
                        if (m_pmesh)
                        {
                            delete m_pmesh;
                            m_pmesh = nullptr;
                        }
                    }
                }
            }
            else
            {
                MeshLib::MeshSubdivision<Real> subdiv_(submesh);
                m_pmesh = subdiv_.SplitQuad();
            }
        }
        else if (submesh->is_quad())
        {
            m_pmesh = submesh;
        }

        bool find_duplicate_or_skip = false;
        if (m_pmesh)
        {
            MySortedTuple<ptrdiff_t, 11, false> info_tuple;
            TinyVector<double, 3> size;
            get_meshinfo(m_pmesh, info_tuple, size);

            auto max_side = std::max({fabs(size[0]), fabs(size[1]), fabs(size[2])});
            auto min_side = std::min({fabs(size[0]), fabs(size[1]), fabs(size[2])});
            if (min_side / (max_side + 1.0e-12) < flat_threshold || quick_filter(m_pmesh))
            {
                if (m_pmesh != submesh)
                {
                    delete m_pmesh;
                    m_pmesh = 0;
                }
                find_duplicate_or_skip = true;
            }
            else
            {
                auto it = database.find(info_tuple);
                if (it == database.end())
                {
                    std::vector<TinyVector<double, 3>> mesh_info;
                    mesh_info.emplace_back(size);
                    database[info_tuple] = mesh_info;
                }
                else
                {
                    auto &bbox_info = it->second;
                    for (const auto &msize : bbox_info)
                    {
                        auto diff = msize - size;
                        auto max_diff = std::max({fabs(diff[0]), fabs(diff[1]), fabs(diff[2])});
                        auto max_side = std::max({fabs(msize[0]), fabs(msize[1]), fabs(msize[2])}) + 1.0e-12;
                        if (max_diff / max_side < bound_threshold)
                        {
                            find_duplicate_or_skip = true;
                            if (m_pmesh != submesh)
                            {
                                delete m_pmesh;
                                m_pmesh = 0;
                            }
                            break;
                        }
                    }
                    if (!find_duplicate_or_skip)
                    {
                        bbox_info.emplace_back(size);
                    }
                }
            }
        }

        if (m_pmesh)
            intermediate_meshes.insert(m_pmesh);
        // bool skip_flag = (!m_pmesh) || m_pmesh->is_quad() == false || (!CheckMeshValidity(m_pmesh));
        bool skip_flag = (!m_pmesh) || (debug == false && m_pmesh->is_quad() == false) || find_duplicate_or_skip;

        if (!skip_flag)
        {
            // check loop quality
            MeshLib::Mesh3D<Real> *subdiv_mesh = nullptr;
            if (!m_pmesh->is_quad())
            {
                MeshLib::MeshSubdivision<Real> subdiv_(m_pmesh);
                subdiv_mesh = subdiv_.SplitQuad();
                intermediate_meshes.insert(subdiv_mesh);
            }
            else
            {
                subdiv_mesh = m_pmesh;
            }

            if (loop_threshold > 0)
            {
                QuadQuality<Real> quality(subdiv_mesh, false);
                // Real fr = quality.get_simple_faceloop_ratio(), er = quality.get_simple_edgeloop_ratio();
                Real fr = quality.get_simple_faceloop_ratio_new(), er = quality.get_simple_edgeloop_ratio_new();
                Real loop_ratio = std::min(fr, er); // quality.get_simple_score();                
                // std::cout << quality.get_num_of_complex() << std::endl;
                // std::cout<< loop_ratio << " " << loop_threshold << std::endl;
                // Real max_area_ratio, min_area_ratio, mean_area_ratio, min_edge_length;
                // quality.get_complex_distribution(max_area_ratio, min_area_ratio, mean_area_ratio, min_edge_length);
                // if (quality.get_num_of_complex() > 1024 || fr < loop_threshold ||min_area_ratio < (Real)1.0/1024 || min_edge_length < (Real)1.0/32) 
                // {
                //     skip_flag = true;
                // }
                // std::cout << min_area_ratio << " " << (Real)1.0/1024 << std::endl;
                // std::cout << min_edge_length << " " << (Real)1.0/32 << std::endl;
                // std::cout << (skip_flag ? "skip" : "accept") << std::endl;
                // if (loop_ratio < loop_threshold) //|| quality.get_num_of_complex() > 4096)
                    // skip_flag = true;
            }
        }

        if (!skip_flag)
        {
            if (debug == false || (debug && !m_pmesh->is_quad()))
            {
                // std::cout << "**********************" << std::endl;
                for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); ++i)
                {
                    auto vert = m_pmesh->get_vertex(i);
                    merged_mesh->insert_vertex(vert->pos);
                }

                for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); ++i)
                {
                    auto face = m_pmesh->get_face(i);
                    std::vector<MeshLib::HE_vert<Real> *> face_vert;
                    face_vert.reserve(face->valence);
                    auto he = face->edge;
                    do
                    {
                        face_vert.emplace_back(merged_mesh->get_vertex(he->vert->id + num_vertices));
                        he = he->next;
                    } while (he != face->edge);
                    merged_mesh->insert_face(face_vert);
                }
                num_vertices += m_pmesh->get_num_of_vertices();
            }
        }

        for (auto imesh : intermediate_meshes)
        {
            if (imesh)
                delete imesh;
        }
    }

    if (merged_mesh->get_num_of_faces() <= min_face_num)
    {
        delete merged_mesh;
        return nullptr;
    }
    else
        return merged_mesh;
}
//////////////////////////////////////////////////////////////////////////
template bool CheckMeshValidity(MeshLib::Mesh3D<double> *mesh);
template MeshLib::Mesh3D<double> *LoopQuadProcessing(MeshLib::Mesh3D<double> *input_mesh, const double loop_threshold, bool debug, bool subdiv, int min_face_num);