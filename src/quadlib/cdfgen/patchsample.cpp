
#include "patchsample.h"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "MeshSubdivision.h"
#include "libnpy/npz.h"
#include "libnpy/npy.h"
#include "PoissonSampling.h"
#include "looputil.h"
#include "PointUtil.h"
#include "myutils.h"
#include <stdexcept>

template <typename Real>
PatchSample<Real>::PatchSample(MeshLib::Mesh3D<Real> *input_quad_mesh,
                               int num_samples,
                               int num_fps_points_min,
                               int num_fps_points_max,
                               int num_fps_copies,
                               bool model_normalize,
                               int random_seed,
                               bool use_mesh_as_complex,
                               Real sharpangle,
                               int resolution,
                               bool debug)
    : quad_mesh(input_quad_mesh), debug_mode(debug)
{
    if (input_quad_mesh == 0 || input_quad_mesh->is_quad() == false)
    {
        std::cout << "Error: the input is not valid!" << std::endl;
        valid = false;
        return;
    }

    valid = !has_zero_length_edge(input_quad_mesh, (Real)1.0e-12);
    if (!valid)
    {
        return;
    }

    if (model_normalize)
    {
        scale_and_PCA(quad_mesh, inverse_rotation, normalization_center, normalization_scale);
    }
    else
    {
        inverse_rotation[0][0] = inverse_rotation[1][1] = inverse_rotation[2][2] = 1;
    }

    // (2) compute patch distance
    compute_patch_distance(use_mesh_as_complex, sharpangle);
    // (3) decompose the quad mesh into a triangle mesh
    decompose_subdiv_mesh();

    // (4) sample points
    std::vector<TinyVector<Real, 3>> vertices(subdiv_mesh->get_num_of_vertices());
    std::vector<ptrdiff_t> triangles(subdiv_mesh->get_num_of_faces() * 6);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
    {
        auto vert = subdiv_mesh->get_vertex(i);
        vertices[i] = vert->pos;
    }
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
    {
        auto face = subdiv_mesh->get_face(i);
        auto edge = face->edge;
        auto qv0 = edge->pair->vert, qv1 = edge->vert, qv2 = edge->next->vert, qv3 = edge->next->next->vert;

        // make the diagonal edge appear last
        if (!split_flip[i])
        {
            triangles[6 * i] = qv0->id, triangles[6 * i + 1] = qv1->id, triangles[6 * i + 2] = qv2->id;
            triangles[6 * i + 3] = qv2->id, triangles[6 * i + 4] = qv3->id, triangles[6 * i + 5] = qv0->id;
        }
        else
        {
            triangles[6 * i] = qv3->id, triangles[6 * i + 1] = qv0->id, triangles[6 * i + 2] = qv1->id;
            triangles[6 * i + 3] = qv1->id, triangles[6 * i + 4] = qv2->id, triangles[6 * i + 5] = qv3->id;
        }
    }

    auto size = std::max(quad_mesh->zmax - quad_mesh->zmin, std::max(quad_mesh->ymax - quad_mesh->ymin, quad_mesh->xmax - quad_mesh->xmin));
    auto radius = resolution > 0 ? (Real)size / (2 * resolution) : (Real)-1;

    PoissonSampling<Real> poisson_sampling(random_seed);

    poisson_sampling.sampling(radius, vertices, triangles,
                              sample_points, sample_point_face_ids,
                              &sample_point_bary_coords, num_samples);

    // apply fps samples
    for (int i = num_fps_points_min; i <= num_fps_points_max; i *= 2)
    {
        num_fps_points_list.push_back(i);
    }

    for (auto &num_fps_points : num_fps_points_list)
    {
        std::vector<std::vector<TinyVector<Real, 3>>> fps_points;
        std::vector<std::vector<TinyVector<Real, 3>>> fps_bary_coords;
        std::vector<std::vector<ptrdiff_t>> fps_point_face_ids;
        int buffer_size = std::max(num_fps_points / 10, 32);
        fps_bary_coords.resize(num_fps_copies);
        fps_point_face_ids.resize(num_fps_copies);
        fps_points.resize(num_fps_copies);
        for (int i = 0; i < num_fps_copies; i++)
        {
            fps_bary_coords[i].reserve(num_fps_points + 2 * buffer_size);
            fps_point_face_ids[i].reserve(num_fps_points + 2 * buffer_size);
            fps_points[i].reserve(num_fps_points + 2 * buffer_size);
        }

        for (int i = 0; i < num_fps_copies; i++)
        {
            bool suc = true;
            do
            {
                suc = false;
                PoissonSampling<Real> poisson_sampling(rand());
                poisson_sampling.sampling((Real)-1, vertices, triangles,
                                          fps_points[i], fps_point_face_ids[i], &fps_bary_coords[i], num_fps_points + buffer_size);
                if (fps_points[i].size() >= num_fps_points)
                {
                    std::vector<size_t> indices(fps_point_face_ids[i].size());
                    std::iota(indices.begin(), indices.end(), 0);
                    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
                    for (size_t j = 0; j < indices.size(); ++j)
                    {
                        std::swap(fps_points[i][j], fps_points[i][indices[j]]);
                        std::swap(fps_bary_coords[i][j], fps_bary_coords[i][indices[j]]);
                        std::swap(fps_point_face_ids[i][j], fps_point_face_ids[i][indices[j]]);
                    }
                    fps_points[i].resize(num_fps_points);
                    fps_point_face_ids[i].resize(num_fps_points);
                    fps_bary_coords[i].resize(num_fps_points);
                    suc = true;
                }
                else
                {
                    // unlikely to happen
                    // std::cout << "FPS sampling failed, retrying..." << std::endl;
                    buffer_size *= 2;
                }
            } while (!suc);
        }
        fps_points_mp[num_fps_points] = fps_points;
        fps_bary_coords_mp[num_fps_points] = fps_bary_coords;
        fps_point_face_ids_mp[num_fps_points] = fps_point_face_ids;
    }

    if (debug_mode)
    {
        std::vector<TinyVector<Real, 3>> sample_grads_0(sample_points.size()), sample_grads_1(sample_points.size());
        std::vector<Real> sample_point_colors_0(sample_points.size()), sample_point_colors_1(sample_points.size()), triangles_colors_0(triangles.size() / 3), triangles_colors_1(triangles.size() / 3);
        std::vector<TinyVector<Real, 3>> sample_normals(sample_points.size());
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < (ptrdiff_t)sample_points.size(); i++)
        {
            Real c0, c1;
            TinyVector<Real, 3> g0, g1;
            get_grading_color_all((int)sample_point_face_ids[i], sample_points[i], c0, c1, &g0, &g1);
            sample_point_colors_0[i] = std::min(c0, c1);
            sample_point_colors_1[i] = std::min(1 - c0, 1 - c1);
            if (c0 < c1)
            {
                sample_grads_0[i] = g0;
                sample_grads_1[i] = -g1;
            }
            else
            {
                sample_grads_0[i] = g1;
                sample_grads_1[i] = -g0;
            }
            sample_grads_0[i].Normalize();
            sample_grads_1[i].Normalize();
            sample_normals[i] = subdiv_mesh->get_face(sample_point_face_ids[i] / 2)->normal;
        }
#pragma omp parallel for
        for (int i = 0; i < (int)triangles.size(); i += 3)
        {
            Real c0, c1;
            get_grading_color_all(i / 3, (vertices[triangles[i]] + vertices[triangles[i + 2]]) / 2, c0, c1);
            triangles_colors_0[i / 3] = std::min(c0, c1);
            triangles_colors_1[i / 3] = std::min(1 - c0, 1 - c1);
        }

        // SavePLYmesh_with_float_storage_and_gray_color("tri_mesh_dcdf.ply", vertices, triangles, &triangles_colors_0);
        // SavePLYmesh_with_float_storage_and_gray_color("tri_mesh_cdf.ply", vertices, triangles, &triangles_colors_1);
        // SavePtsPLY("sample_points_dcdf.ply", sample_points, &sample_grads_0, &sample_point_colors_0);
        // SavePtsPLY("sample_points_cdf.ply", sample_points, &sample_grads_1, &sample_point_colors_1);
        SavePtsPLY("sample_points_dcdf.ply", sample_points, &sample_normals, &sample_point_colors_0);
        SavePtsPLY("sample_points_cdf.ply", sample_points, &sample_normals, &sample_point_colors_1);
        // for (auto &num_fps_points : num_fps_points_list)
        // {
        //     auto &fps_points = fps_points_mp[num_fps_points];
        //     const auto &fps_point_face_ids = fps_point_face_ids_mp[num_fps_points];
        //     std::vector<Real> fps_points_colors(fps_points[0].size());
        //     for (int i = 0; i < num_fps_copies; i++)
        //     {
        //         // #pragma omp parallel for
        //         for (ptrdiff_t j = 0; j < (ptrdiff_t)fps_points[i].size(); j++)
        //         {
        //             fps_points_colors[j] = get_grading_color(fps_point_face_ids[i][j], fps_points[i][j]);
        //         }
        //         SavePtsPLY("fps_points_" + std::to_string(num_fps_points) + "_" + std::to_string(i) + ".ply", fps_points[i], (std::vector<TinyVector<Real, 3>> *)0, &fps_points_colors, false);
        //     }
        // }
    }
}
////////////////////////////////////////////////
template <typename Real>
PatchSample<Real>::~PatchSample()
{
    if (subdiv_mesh != 0)
        delete subdiv_mesh;
    if (dualquad_mesh)
        delete dualquad_mesh;
}
////////////////////////////////////////////////
template <typename Real>
void PatchSample<Real>::save_samples_to_npz(const std::string &npzfilename, bool quadextractioninfo)
{
    if (!valid)
    {
        // std::cout << "Error: the input is not valid!" << std::endl;
        return;
    }

    npy::onpzstream output(npzfilename);

    std::vector<size_t> divshape({1, 1});
    npy::tensor<std::int8_t> is_checkerboard_tensor(divshape);
    is_checkerboard_tensor(0, 0) = 2;
    output.write("is_checkerboard", is_checkerboard_tensor); // compatibility with old versions

    npy::tensor<std::uint32_t> nc(divshape);
    // npy::tensor<std::uint32_t> ns(divshape);
    npy::tensor<std::uint32_t> nb(divshape);
    // npy::tensor<float> faceloopscore(divshape);
    nc(0, 0) = static_cast<std::uint32_t>(num_complex);
    // ns(0, 0) = static_cast<std::uint32_t>(num_singularity);
    nb(0, 0) = static_cast<std::uint32_t>(num_boundary);
    // faceloopscore(0, 0) = static_cast<float>(loop_score);
    output.write("num_complex", nc);
    // output.write("num_singularity", ns);
    output.write("num_boundary", nb);
    // output.write("face_loop_score", faceloopscore);

    // export normalization information
    std::vector<size_t> matrixsize({3, 4});
    npy::tensor<float> invT(matrixsize);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            invT(i, j) = static_cast<float>(inverse_rotation[i][j] * normalization_scale);
        }
        invT(i, 3) = static_cast<float>(normalization_center[i]);
    }
    output.write("invT", invT);

    // std::vector<TinyVector<Real, 3>> quad_mesh_vertices(quad_mesh->get_num_of_vertices());
    // for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    // {
    //     auto vert = quad_mesh->get_vertex(i);
    //     quad_mesh_vertices[i][0] = normalization_scale * (inverse_rotation[0] * vert->pos) + normalization_center[0];
    //     quad_mesh_vertices[i][1] = normalization_scale * (inverse_rotation[1] * vert->pos) + normalization_center[1];
    //     quad_mesh_vertices[i][2] = normalization_scale * (inverse_rotation[2] * vert->pos) + normalization_center[2];
    // }
    // SavePtsPLY("quad_mesh.ply", quad_mesh_vertices, (std::vector<TinyVector<Real, 3>> *)0, (std::vector<Real> *)0, false);

    // original quad mesh
    npy::tensor<std::uint32_t> quad_vertex_num(divshape);
    quad_vertex_num(0, 0) = static_cast<std::uint32_t>(quad_mesh->get_num_of_vertices());

    std::vector<size_t> quadshape({(size_t)quad_mesh->get_num_of_faces(), 4});
    npy::tensor<std::uint32_t> quad_facet(quadshape);
#pragma omp parallel for
    for (int i = 0; i < (int)quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        auto edge = face->edge;
        for (int j = 0; j < 4; j++)
        {
            quad_facet(i, j) = static_cast<std::uint32_t>(edge->pair->vert->id);
            edge = edge->next;
        }
    }
    output.write("quad_mesh_vertex_num", quad_vertex_num);
    output.write("quad_facet", quad_facet);

    // suddiv mesh
    size_t subdiv_nf = (size_t)subdiv_mesh->get_num_of_faces(), subdiv_nv = (size_t)subdiv_mesh->get_num_of_vertices();
    std::vector<size_t> subdiv_vertexshape({subdiv_nv, 3}), subdiv_facetshape({subdiv_nf, 4}), quadinfoshape({subdiv_nf, 1}), quad2checkeridshape({quad2patches.size(), 1});
    npy::tensor<float> subdiv_vertex(subdiv_vertexshape);
    npy::tensor<std::uint32_t> subdiv_facet(subdiv_facetshape), face2quad(quadinfoshape), quad2patch(quad2checkeridshape);
    npy::tensor<std::int8_t> quad_split(quadinfoshape);
#pragma omp parallel for
    for (int i = 0; i < (int)subdiv_mesh->get_num_of_vertices(); i++)
    {
        auto vert = subdiv_mesh->get_vertex(i);
        subdiv_vertex(i, 0) = static_cast<float>(vert->pos[0]), subdiv_vertex(i, 1) = static_cast<float>(vert->pos[1]), subdiv_vertex(i, 2) = static_cast<float>(vert->pos[2]);
    }
#pragma omp parallel for
    for (int i = 0; i < (int)subdiv_mesh->get_num_of_faces(); i++)
    {
        auto face = subdiv_mesh->get_face(i);
        auto edge = face->edge;
        for (int j = 0; j < 4; j++)
        {
            subdiv_facet(i, j) = static_cast<std::uint32_t>(edge->pair->vert->id);
            edge = edge->next;
        }
        face2quad(i, 0) = static_cast<std::uint32_t>(quad_id_map[i]);
        quad_split(i, 0) = static_cast<std::int8_t>(split_flip[i]);
    }
#pragma omp parallel for
    for (int i = 0; i < (int)quad2patches.size(); i++)
    {
        quad2patch(i, 0) = static_cast<std::uint32_t>(quad2patches[i]);
    }

    output.write("subdiv_vertex", subdiv_vertex);
    output.write("subdiv_facet", subdiv_facet);
    output.write("face2quad", face2quad);
    output.write("quad2patch", quad2patch);
    output.write("quad_split", quad_split);

    //////////////////////////////////
    // export quad faces
    // center/normal/colors/checker ids/offset-ids
    size_t nf = (size_t)dualquad_mesh->get_num_of_faces();
    std::vector<size_t> dualquadoffsetshape3({nf, 3}), dualquadoffsetshape4({nf, 4});
    npy::tensor<std::uint32_t> offset_abcd_ids(dualquadoffsetshape4), offset_123_ids(dualquadoffsetshape3);

    std::vector<int> boundary_vertex_tag(dualquad_mesh->get_num_of_vertices(), 0);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < dualquad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = dualquad_mesh->get_vertex(i);
        boundary_vertex_tag[i] = dualquad_mesh->is_on_boundary(hv) ? 1 : 0;
    }

    std::vector<ptrdiff_t> dualquad_face_checker_vertex_id(dualquad_mesh->get_num_of_faces(), -1);
#pragma omp parallel for
    for (int i = 0; i < (int)dualquad_mesh->get_num_of_faces(); i++)
    {
        auto face = dualquad_mesh->get_face(i);
        auto edge = face->edge;
        int vindex = -1;
        do
        {
            if (edge->vert->tag)
            {
                dualquad_face_checker_vertex_id[i] = dualquad_vertex_id_map[edge->vert->id];
                break;
            }
            edge = edge->next;
        } while (edge != face->edge);
    }

#pragma omp parallel for
    for (int i = 0; i < (int)dualquad_mesh->get_num_of_faces(); i++)
    {
        auto face = dualquad_mesh->get_face(i);
        auto edge = face->edge;
        auto start_edge = edge->next;
        std::vector<MeshLib::HE_vert<Real> *> fvert = {edge->vert, edge->next->vert, edge->next->next->vert, edge->next->next->next->vert};
        int vindex = -1;
        for (int j = 0; j < 4; j++)
        {
            if (fvert[j]->tag)
            {
                vindex = j;
                break;
            }
            start_edge = start_edge->next;
        }

        offset_abcd_ids(i, 0) = static_cast<std::uint32_t>(dualquad_vertex_id_map[fvert[vindex]->id]);
        offset_abcd_ids(i, 1) = static_cast<std::uint32_t>(dualquad_vertex_id_map[fvert[(vindex + 1) % 4]->id]);
        offset_abcd_ids(i, 2) = static_cast<std::uint32_t>(dualquad_vertex_id_map[fvert[(vindex + 2) % 4]->id]);
        offset_abcd_ids(i, 3) = static_cast<std::uint32_t>(dualquad_vertex_id_map[fvert[(vindex + 3) % 4]->id]);

        start_edge = start_edge->next;

        offset_123_ids(i, 0) = static_cast<std::uint32_t>(start_edge->pair->face ? dualquad_face_checker_vertex_id[start_edge->pair->face->id] : dualquad_vertex_id_map[start_edge->pair->vert->id]);
        offset_123_ids(i, 1) = static_cast<std::uint32_t>(boundary_vertex_tag[start_edge->vert->id] ? dualquad_vertex_id_map[start_edge->vert->id] : dualquad_face_checker_vertex_id[start_edge->pair->prev->pair->face->id]);

        start_edge = start_edge->next;

        offset_123_ids(i, 2) = static_cast<std::uint32_t>(start_edge->pair->face ? dualquad_face_checker_vertex_id[start_edge->pair->face->id] : dualquad_vertex_id_map[start_edge->pair->vert->id]);
    }
    output.write("offset_abcd_id", offset_abcd_ids);
    output.write("offset_123_id", offset_123_ids);

    // export quad face center, normal, color, checker id
    // export color information
    std::vector<size_t> colorstoresize({edge_color_store.size(), 2}), edgecolorsize({(size_t)subdiv_mesh->get_num_of_faces() * 4, 1});
    npy::tensor<float> colorstore(colorstoresize);
    npy::tensor<std::int32_t> edge_color(edgecolorsize);
#pragma omp parallel for
    for (int i = 0; i < edge_color_store.size(); i++)
    {
        colorstore(i, 0) = static_cast<float>(edge_color_store[i].first), colorstore(i, 1) = static_cast<float>(edge_color_store[i].second);
    }

#pragma omp parallel for
    for (int i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
    {
        auto edge = subdiv_mesh->get_face(i)->edge;
        edge_color(4 * i, 0) = static_cast<std::int32_t>(edge_vertex_color[edge->id]);
        edge_color(4 * i + 1, 0) = static_cast<std::int32_t>(edge_vertex_color[edge->next->id]);
        edge_color(4 * i + 2, 0) = static_cast<std::int32_t>(edge_vertex_color[edge->next->next->id]);
        edge_color(4 * i + 3, 0) = static_cast<std::int32_t>(edge_vertex_color[edge->next->next->next->id]);
    }
    output.write("edge_vertex_color", edge_color);
    output.write("edge_color_store", colorstore);

    //////////////////////////////////

    // edge information
    std::vector<size_t> edgeshape({(size_t)subdiv_mesh->get_num_of_edges() / 2, 2}), vertshape({(size_t)subdiv_mesh->get_num_of_edges() / 2, 4});
    npy::tensor<std::uint32_t> edge_faceids(edgeshape); // face0, face1: triface
    npy::tensor<std::int8_t> edge_info(vertshape);      // quad_face_edge_id, opp_vert_local_id, tag;  opp_vert: opposite vert at neighbor tri face, tag: 1: border, 0: non-border

    std::vector<ptrdiff_t> neighbortrifaceid(subdiv_mesh->get_num_of_edges(), -1);
    std::vector<int> local_edge_indices(subdiv_mesh->get_num_of_edges(), -1);
    std::vector<int> local_opp_indices(subdiv_mesh->get_num_of_edges(), -1);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
    {
        auto face = subdiv_mesh->get_face(i);
        auto edge = face->edge;
        auto qv0 = edge->pair->vert, qv1 = edge->vert, qv2 = edge->next->vert, qv3 = edge->next->next->vert;

        local_edge_indices[edge->id] = 0;
        local_edge_indices[edge->next->id] = 1;
        local_edge_indices[edge->next->next->id] = 2;
        local_edge_indices[edge->next->next->next->id] = 3;

        if (!split_flip[i])
        {
            neighbortrifaceid[edge->id] = neighbortrifaceid[edge->next->id] = 2 * i;
            neighbortrifaceid[edge->next->next->id] = neighbortrifaceid[edge->next->next->next->id] = 2 * i + 1;
            local_opp_indices[edge->id] = 2;
            local_opp_indices[edge->next->id] = 0;
            local_opp_indices[edge->next->next->id] = 2;
            local_opp_indices[edge->next->next->next->id] = 0;
        }
        else
        {
            neighbortrifaceid[edge->id] = neighbortrifaceid[edge->next->next->next->id] = 2 * i;
            neighbortrifaceid[edge->next->id] = neighbortrifaceid[edge->next->next->id] = 2 * i + 1;
            local_opp_indices[edge->id] = 0;
            local_opp_indices[edge->next->id] = 2;
            local_opp_indices[edge->next->next->id] = 0;
            local_opp_indices[edge->next->next->next->id] = 2;
        }
    }

    int edge_counter = 0;
    for (int i = 0; i < (int)subdiv_mesh->get_num_of_edges(); i++)
    {
        auto edge = subdiv_mesh->get_edge(i);
        if (edge->id > edge->pair->id || edge->face == 0)
            continue;

        edge_faceids(edge_counter, 0) = neighbortrifaceid[edge->id] != -1 ? static_cast<std::uint32_t>(neighbortrifaceid[edge->id]) : static_cast<std::uint32_t>(neighbortrifaceid[edge->pair->id]);
        edge_faceids(edge_counter, 1) = neighbortrifaceid[edge->pair->id] != -1 ? static_cast<std::uint32_t>(neighbortrifaceid[edge->pair->id]) : static_cast<std::uint32_t>(neighbortrifaceid[edge->id]);

        const auto &local_edge_index = edge->face ? local_edge_indices[edge->id] : local_edge_indices[edge->pair->id];
        const auto &local_opp_index = edge->pair->face ? local_opp_indices[edge->pair->id] : local_opp_indices[edge->id];
        const auto &local_opp_index2 = edge->face ? local_opp_indices[edge->id] : local_opp_indices[edge->pair->id];

        bool border = edge->face && edge->pair->face && quad2patches[quad_id_map[edge->pair->face->id]] != quad2patches[quad_id_map[edge->face->id]];

        edge_info(edge_counter, 0) = static_cast<std::int8_t>(local_edge_index);
        edge_info(edge_counter, 1) = static_cast<std::int8_t>(local_opp_index);
        edge_info(edge_counter, 2) = static_cast<std::int8_t>(local_opp_index2);
        edge_info(edge_counter, 3) = static_cast<std::int8_t>(border);

        edge_counter++;
    }
    output.write("edge_faceids", edge_faceids);
    output.write("edge_info", edge_info);

    // sample points
    std::vector<size_t> sampleshape({sample_points.size(), 3}), sample_colorshape({sample_points.size(), 1});
    npy::tensor<float> samplepts(sampleshape), sample_bary_coords(sampleshape);
    npy::tensor<std::uint32_t> point2triid(sample_colorshape);
#pragma omp parallel for
    for (int i = 0; i < (int)sample_points.size(); i++)
    {
        samplepts(i, 0) = static_cast<float>(sample_points[i][0]), samplepts(i, 1) = static_cast<float>(sample_points[i][1]), samplepts(i, 2) = static_cast<float>(sample_points[i][2]);
        sample_bary_coords(i, 0) = static_cast<float>(sample_point_bary_coords[i][0]), sample_bary_coords(i, 1) = static_cast<float>(sample_point_bary_coords[i][1]), sample_bary_coords(i, 2) = static_cast<float>(sample_point_bary_coords[i][2]);
        auto findex = sample_point_face_ids[i];
        point2triid(i, 0) = static_cast<std::uint32_t>(findex);
    }
    output.write("sample_bary_coords", sample_bary_coords);
    output.write("sample_point_tri_id", point2triid);

    // export fps samples
    for (auto &num_fps_points : num_fps_points_list)
    {
        const auto &fps_bary_coords = fps_bary_coords_mp[num_fps_points];
        const auto &fps_point_face_ids = fps_point_face_ids_mp[num_fps_points];
        const auto &fps_points = fps_points_mp[num_fps_points];

        std::vector<size_t> fpsshape({fps_bary_coords.size(), fps_bary_coords[0].size(), 3}), fps_face_ids_shape({fps_bary_coords.size(), fps_bary_coords[0].size(), 1});
        npy::tensor<float> fps_points_tensor(fpsshape);
        npy::tensor<std::uint32_t> fps_face_ids(fps_face_ids_shape);
#pragma omp parallel for
        for (int i = 0; i < (int)fps_bary_coords.size(); i++)
        {
            for (int j = 0; j < (int)fps_bary_coords[i].size(); j++)
            {
                fps_points_tensor(i, j, 0) = static_cast<float>(fps_bary_coords[i][j][0]);
                fps_points_tensor(i, j, 1) = static_cast<float>(fps_bary_coords[i][j][1]);
                fps_points_tensor(i, j, 2) = static_cast<float>(fps_bary_coords[i][j][2]);
                fps_face_ids(i, j, 0) = static_cast<std::uint32_t>(fps_point_face_ids[i][j]);
            }
        }
        output.write("fps_bary_coords_" + std::to_string(num_fps_points), fps_points_tensor);
        output.write("fps_point_tri_id_" + std::to_string(num_fps_points), fps_face_ids);
    }

    // export information for quad extraction
    if (quadextractioninfo)
    {
        MeshLib::Mesh3D<Real> *trimesh = new MeshLib::Mesh3D<Real>;
        for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
        {
            trimesh->insert_vertex(subdiv_mesh->get_vertex(i)->pos);
        }
        std::vector<MeshLib::HE_vert<Real> *> facelist;
        for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
        {
            auto face = subdiv_mesh->get_face(i);
            auto edge = face->edge;
            auto qv0 = edge->pair->vert, qv1 = edge->vert, qv2 = edge->next->vert, qv3 = edge->next->next->vert;

            // make the diagonal edge appear last
            if (!split_flip[i])
            {
                facelist = {trimesh->get_vertex(qv0->id), trimesh->get_vertex(qv1->id), trimesh->get_vertex(qv2->id)};
                trimesh->insert_face(facelist);
                facelist = {trimesh->get_vertex(qv2->id), trimesh->get_vertex(qv3->id), trimesh->get_vertex(qv0->id)};
                trimesh->insert_face(facelist);
            }
            else
            {
                facelist = {trimesh->get_vertex(qv3->id), trimesh->get_vertex(qv0->id), trimesh->get_vertex(qv1->id)};
                trimesh->insert_face(facelist);
                facelist = {trimesh->get_vertex(qv1->id), trimesh->get_vertex(qv2->id), trimesh->get_vertex(qv3->id)};
                trimesh->insert_face(facelist);
            }
        }
        trimesh->update_mesh();

        int num_subdiv = 2;
        for (int iter = 0; iter < num_subdiv; iter++)
        {
            MeshLib::MeshSubdivision<Real> subdiv_(trimesh);
            auto tmp_mesh = subdiv_.SplitTri();
            std::swap(trimesh, tmp_mesh);
            delete tmp_mesh;
        }

        int div = (int)pow(4, num_subdiv);
        size_t nf = trimesh->get_num_of_faces();
        std::vector<size_t> facecolorshape({nf, 1}), offsetshape({nf, 3});
        npy::tensor<float> cdf_color(facecolorshape), dcdf_color(facecolorshape), offset(offsetshape), offsetc(offsetshape), offsetb(offsetshape), offsetd(offsetshape);
        std::vector<TinyVector<Real, 3>> cdf_gradients(nf), dcdf_gradients(nf);
#pragma omp parallel for
        for (int i = 0; i < (int)trimesh->get_num_of_faces(); i++)
        {
            auto face = trimesh->get_face(i);
            auto edge = face->edge;
            auto pos = face->GetCentroid();
            Real c0, c1;
            TinyVector<Real, 3> grad0, grad1;
            get_grading_color_all(i / div, pos, c0, c1, &grad0, &grad1);
            dcdf_color(i, 0) = static_cast<float>(std::min(c0, c1));
            cdf_color(i, 0) = static_cast<float>(std::min(1 - c0, 1 - c1));
            dcdf_gradients[i] = (c0 < c1) ? grad0 : grad1;
            dcdf_gradients[i].Normalize();
            cdf_gradients[i] = (c0 < c1) ? -grad1 : -grad0;
            cdf_gradients[i].Normalize();
            const auto &center = subdiv_mesh->get_vertex(offset_abcd_ids((int)face2quad(i / (2 * div), 0), 0))->pos;
            const auto &ccenter = subdiv_mesh->get_vertex(offset_abcd_ids((int)face2quad(i / (2 * div), 0), 2))->pos;
            const auto &bcenter = subdiv_mesh->get_vertex(offset_abcd_ids((int)face2quad(i / (2 * div), 0), 1))->pos;
            const auto &dcenter = subdiv_mesh->get_vertex(offset_abcd_ids((int)face2quad(i / (2 * div), 0), 3))->pos;
            auto doffset = center - pos;
            auto coffest = ccenter - pos;
            auto boffset = bcenter - pos;
            auto doffsetd = dcenter - pos;
            offset(i, 0) = static_cast<float>(doffset[0]), offset(i, 1) = static_cast<float>(doffset[1]), offset(i, 2) = static_cast<float>(doffset[2]);
            offsetc(i, 0) = static_cast<float>(coffest[0]), offsetc(i, 1) = static_cast<float>(coffest[1]), offsetc(i, 2) = static_cast<float>(coffest[2]);
            offsetb(i, 0) = static_cast<float>(boffset[0]), offsetb(i, 1) = static_cast<float>(boffset[1]), offsetb(i, 2) = static_cast<float>(boffset[2]);
            offsetd(i, 0) = static_cast<float>(doffsetd[0]), offsetd(i, 1) = static_cast<float>(doffsetd[1]), offsetd(i, 2) = static_cast<float>(doffsetd[2]);
        }

        auto dir = GetFileDirectory(npzfilename);
        std::filesystem::path npzpath(npzfilename);
        std::filesystem::path subdiv_filename = npzpath.parent_path() / "subdiv.ply";
        save_mesh(trimesh, subdiv_filename.string());

        output.write("cdf", cdf_color);
        output.write("dcdf", dcdf_color);
        output.write("offsetc", offsetc);
        output.write("offset", offset);
        output.write("offsetb", offsetb);
        output.write("offsetd", offsetd);

        // std::filesystem::path uvobj_filename = npzpath.parent_path() / "subdivuv.obj";
        // std::ofstream output_uvobj(uvobj_filename.string());
        // for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
        //     output_uvobj << "v " << subdiv_mesh->get_vertex(i)->pos << "\n";
        // std::vector<bool> vertex_tag(subdiv_mesh->get_num_of_vertices(), false);
        // for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_edges(); i++)
        // {
        //     edge_color_store[]
        // }

        // std::filesystem::path rosy_filename = npzpath.parent_path() / "subdiv.rosy";
        // std::ofstream output_rosy(rosy_filename.string());
        // output_rosy << cdf_gradients.size() << "\n4\n";
        // for (int i = 0; i < (int)cdf_gradients.size(); i++)
        // {
        //     output_rosy << cdf_gradients[i] << "\n";
        // }
        // output_rosy.close();
        // std::filesystem::path frame_filename = npzpath.parent_path() / "subdiv.frame";
        // std::ofstream output_frame(frame_filename.string());
        // output_frame << "4 " << cdf_gradients.size() << "\n";
        // for (int i = 0; i < (int)cdf_gradients.size(); i++)
        // {
        //     output_frame << i << ' ' << cdf_gradients[i] << ' ' << dcdf_gradients[i] << "\n";
        // }
        // output_frame.close();

        // std::filesystem::path uvobj_filename = npzpath.parent_path() / "subdiv.obj";
        // std::ofstream output_uvobj(uvobj_filename.string());
        // for (ptrdiff_t i = 0; i < trimesh->get_num_of_vertices(); i++)
        // {
        //     auto v = trimesh->get_vertex(i);
        //     output_uvobj << "v " << v->pos[0] << ' ' << v->pos[1] << ' ' << v->pos[2] << "\n";
        // }
        // Real c0, c1, u, v;
        // size_t count = 1;
        // for (ptrdiff_t i = 0; i < trimesh->get_num_of_faces(); i++)
        // {
        //     auto face = trimesh->get_face(i);
        //     auto edge = face->edge;
        //     do
        //     {

        //         get_grading_color_all(i / div, edge->vert->pos, c0, c1);
        //         v = static_cast<float>(std::min(c0, c1));
        //         u = 1 - static_cast<float>(std::min(1 - c0, 1 - c1));
        //         output_uvobj << "vt " << u << ' ' << v << "\n";
        //         edge = edge->next;
        //     } while (edge != face->edge);
        //     output_uvobj << "f";
        //     do
        //     {
        //         output_uvobj << ' ' << (edge->vert->id + 1) << '/' << count++;
        //         edge = edge->next;
        //     } while (edge != face->edge);
        //     output_uvobj << "\n";
        // }
        // output_uvobj.close();
        delete trimesh;
    }

    output.close();
}
////////////////////////////////////////////////
template <typename Real>
void PatchSample<Real>::compute_patch_distance(bool use_mesh_as_complex, Real sharpangle)
{
    std::vector<bool> complex_edge_tag, corner_tag;
    std::vector<std::vector<MeshLib::HE_edge<Real> *>> complex_edge_loops;
    std::vector<std::vector<ptrdiff_t>> complex_edge_loops_corner_starting_edges;
    std::unordered_map<complex_arc, complex_arc_info> complex_arcs;
    std::vector<std::vector<ptrdiff_t>> complex_edge_loops_neighbor_cluster_ids;
    std::vector<ptrdiff_t> complex_edge_loops_cluster_ids;
    size_t arc_group_num = 0;

    if (!use_mesh_as_complex)
    {
        BaseComplex<Real> basecomplex(quad_mesh, sharpangle);
        complex_edge_tag = basecomplex.get_complex_edge_tag();
        corner_tag = basecomplex.get_corner_tag();
        complex_edge_loops = basecomplex.get_complex_edge_loops();
        complex_edge_loops_corner_starting_edges = basecomplex.get_complex_edge_loops_corner_starting_edges();
        complex_edge_loops_neighbor_cluster_ids = basecomplex.get_complex_edge_loops_neighbor_cluster_ids();
        complex_edge_loops_cluster_ids = basecomplex.get_complex_edge_loops_cluster_ids();
        complex_arcs = basecomplex.get_complex_arcs();
        arc_group_num = basecomplex.get_arc_group_num();
        num_complex = basecomplex.get_num_complex();
        num_singularity = basecomplex.get_num_singularity();
        if (debug_mode)
        {
            SavePLYmesh_with_float_storage(quad_mesh, "complex_patch.ply", &basecomplex.get_face_patch_ids(), false);
        }
    }
    else
    {
        set_mesh_as_complex(quad_mesh, complex_edge_tag, corner_tag, complex_edge_loops,
                            complex_edge_loops_corner_starting_edges, complex_edge_loops_neighbor_cluster_ids, complex_edge_loops_cluster_ids,
                            complex_arcs, arc_group_num);
        num_complex = (int)quad_mesh->get_num_of_faces();
        num_singularity = (int)quad_mesh->get_num_of_vertices();
    }
    num_boundary = quad_mesh->get_num_of_boundaries();

    if (debug_mode)
    {
        // SavePLYmesh_with_float_storage(quad_mesh, "quad_mesh.ply");
        // SavePLYmesh_with_float_storage(base_complex, "base_complex.ply");

        SaveMarkedEdge_as_ply(quad_mesh, "complex_edge.ply", complex_edge_tag, 0.001f);
        // save_complex_edges("complex_edge.obj", quad_mesh, complex_edge_tag);
    }
    // delete base_complex;

    // (1) split mesh into dual patches

    std::vector<int> boundary_vertex_tag(quad_mesh->get_num_of_vertices(), 0);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = quad_mesh->get_vertex(i);
        boundary_vertex_tag[i] = quad_mesh->is_on_boundary(hv) ? 1 : 0;
    }
    // (1.1) find split point
    std::vector<std::vector<Real>> arc_lengths(arc_group_num);
    std::vector<Real> arc_edge_length;
    arc_edge_length.reserve(quad_mesh->get_num_of_edges() / 2);
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

            if (start_vert->id > end_vert->id && boundary_vertex_tag[start_vert->id] == 0)
                continue;

            complex_arc arc(start_vert->id, end_vert->id, complex_edge_loops_cluster_ids[i], complex_edge_loops_neighbor_cluster_ids[i][j]);

            auto iter = complex_arcs.find(arc);
            auto gid = iter->second.group_id;
            bool order = iter->first.end_vertices[0] == start_vert->id ? iter->second.orientation : !iter->second.orientation;

            arc_edge_length.resize(0);
            Real accum_len = 0;
            for (size_t k = 0; k < boundary_edge_loop.size(); k++)
            {
                auto edge = boundary_edge_loop[(k + corner_indices_on_loop[j]) % boundary_edge_loop.size()];
                if (edge == end_edge)
                {
                    if (arc_lengths[gid].empty())
                    {
                        arc_lengths[gid].assign(arc_edge_length.size(), 0);
                    }
                    Real accum = 0;
                    Real sum_len = std::accumulate(arc_edge_length.begin(), arc_edge_length.end(), (Real)0);
                    if (order)
                    {
                        for (size_t l = 0; l < arc_edge_length.size(); l++)
                        {
                            accum += arc_edge_length[l];
                            arc_lengths[gid][l] += accum / sum_len;
                        }
                    }
                    else
                    {
                        for (int l = (int)arc_edge_length.size() - 1; l >= 0; l--)
                        {
                            accum += arc_edge_length[l];
                            arc_lengths[gid][arc_edge_length.size() - l - 1] += accum / sum_len;
                        }
                    }
                    arc_edge_length.resize(0);
                    break;
                }
                arc_edge_length.emplace_back(edge->GetLength());
            }
        }
    }

    std::vector<int> split_positions(arc_group_num, 0);
    std::vector<Real> interpolation_weights(arc_group_num, 0);
    for (size_t i = 0; i < arc_group_num; i++)
    {
        for (int j = 0; j < (int)arc_lengths[i].size(); j++)
        {
            auto t = arc_lengths[i][j] / arc_lengths[i].back();
            if (t >= (Real)0.5)
            {
                split_positions[i] = j;
                auto denom = (j > 0 ? (arc_lengths[i][j] - arc_lengths[i][j - 1]) : arc_lengths[i][j]);
                interpolation_weights[i] = (arc_lengths[i][j] - arc_lengths[i].back() * (Real)0.5) / denom;
                interpolation_weights[i] = std::max((Real)0.01, std::min((Real)0.99, interpolation_weights[i])); // avoid being too close to 0 or 1
                break;
            }
        }
    }

    // (1.2) create new vertices at complex edges
    subdiv_mesh = new MeshLib::Mesh3D<Real>;
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_vertices(); i++)
    {
        auto vert = quad_mesh->get_vertex(i);
        subdiv_mesh->insert_vertex(vert->pos);
    }
    std::queue<std::pair<MeshLib::HE_edge<Real> *, Real>> edge_queue;
    std::vector<MeshLib::HE_vert<Real> *> edge_to_vertex_map(quad_mesh->get_num_of_edges(), 0);
    std::vector<Real> edge_interpolation_weight(quad_mesh->get_num_of_edges(), 0);

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

            if (start_vert->id > end_vert->id && boundary_vertex_tag[start_vert->id] == 0)
                continue;

            complex_arc arc(start_vert->id, end_vert->id, complex_edge_loops_cluster_ids[i], complex_edge_loops_neighbor_cluster_ids[i][j]);
            auto iter = complex_arcs.find(arc);
            bool order = iter->first.end_vertices[0] == start_vert->id ? iter->second.orientation : !iter->second.orientation;
            auto gid = iter->second.group_id;

            arc_edge_length.resize(0);
            Real accum_len = 0;
            int count = 0;
            for (size_t k = 0; k < boundary_edge_loop.size(); k++)
            {
                auto edge = boundary_edge_loop[(k + corner_indices_on_loop[j]) % boundary_edge_loop.size()];
                if (edge == end_edge)
                {
                    count = 0;
                    break;
                }
                if (order)
                {
                    if (count == split_positions[gid])
                    {
                        auto p = interpolation_weights[gid] * edge->pair->vert->pos + (1 - interpolation_weights[gid]) * edge->vert->pos;
                        auto new_vert = subdiv_mesh->insert_vertex(p);
                        edge_to_vertex_map[edge->id] = edge_to_vertex_map[edge->pair->id] = new_vert;
                        edge_interpolation_weight[edge->id] = interpolation_weights[gid];
                        edge_interpolation_weight[edge->pair->id] = 1 - interpolation_weights[gid];
                        edge_queue.push(std::make_pair(edge, interpolation_weights[gid]));
                        edge_queue.push(std::make_pair(edge->pair, 1 - interpolation_weights[gid]));
                        break;
                    }
                }
                else
                {
                    if (arc_lengths[gid].size() - count - 1 == split_positions[gid])
                    {
                        auto p = interpolation_weights[gid] * edge->vert->pos + (1 - interpolation_weights[gid]) * edge->pair->vert->pos;
                        auto new_vert = subdiv_mesh->insert_vertex(p);
                        edge_to_vertex_map[edge->id] = edge_to_vertex_map[edge->pair->id] = new_vert;
                        edge_interpolation_weight[edge->id] = 1 - interpolation_weights[gid];
                        edge_interpolation_weight[edge->pair->id] = interpolation_weights[gid];
                        edge_queue.push(std::make_pair(edge, 1 - interpolation_weights[gid]));
                        edge_queue.push(std::make_pair(edge->pair, interpolation_weights[gid]));
                        break;
                    }
                }
                count++;
            }
        }
    }

    // {
    //     std::vector<TinyVector<Real, 3>> vertices;
    //     for (ptrdiff_t i = quad_mesh->get_num_of_vertices(); i < subdiv_mesh->get_num_of_vertices(); i++)
    //     {
    //         vertices.push_back(subdiv_mesh->get_vertex(i)->pos);
    //     }
    //     SavePtsPLY("edgep.ply", vertices);
    // }

    // (1.3) create splitted edges at non-complex edges
    while (!edge_queue.empty())
    {
        auto edge = edge_queue.front().first;
        auto t = 1 - edge_queue.front().second;
        edge_queue.pop();
        if (edge->face == 0)
            continue;
        auto opposite_edge = edge->next->next;
        if (edge_to_vertex_map[opposite_edge->id] != 0)
            continue;

        auto p = (1 - t) * opposite_edge->vert->pos + t * opposite_edge->pair->vert->pos;
        auto new_vert = subdiv_mesh->insert_vertex(p);
        edge_to_vertex_map[opposite_edge->id] = edge_to_vertex_map[opposite_edge->pair->id] = new_vert;
        edge_interpolation_weight[opposite_edge->id] = t;
        edge_interpolation_weight[opposite_edge->pair->id] = 1 - t;
        edge_queue.push(std::make_pair(opposite_edge, t));
        edge_queue.push(std::make_pair(opposite_edge->pair, 1 - t));
    }

    // {
    //     std::vector<TinyVector<Real, 3>> vertices;
    //     for (ptrdiff_t i = quad_mesh->get_num_of_vertices(); i < subdiv_mesh->get_num_of_vertices(); i++)
    //     {
    //         vertices.push_back(subdiv_mesh->get_vertex(i)->pos);
    //     }
    //     SavePtsPLY("edge1.ply", vertices);
    // }

    // quad_mesh->write_obj("quad_mesh.obj");
    // return;
    // (1.4) create new faces
    MeshLib::HE_vert<Real> *edge_vertices[4];
    Real edge_weights[4];
    std::vector<MeshLib::HE_vert<Real> *> vertex_list(4);
    ptrdiff_t id[4];
    MeshLib::HE_vert<Real> *original_vertices[4];
    for (ptrdiff_t i = 0; i < quad_mesh->get_num_of_faces(); i++)
    {
        auto face = quad_mesh->get_face(i);
        auto edge = face->edge;
        for (int j = 0; j < 4; j++)
        {
            edge_vertices[j] = edge_to_vertex_map[edge->id];
            edge_weights[j] = edge_interpolation_weight[edge->id];
            id[j] = edge->pair->vert->id;
            original_vertices[j] = subdiv_mesh->get_vertex(id[j]);
            edge = edge->next;
        }

        if (edge_vertices[0] != 0 && edge_vertices[1] != 0)
        {
            auto p0 = edge_weights[0] * edge_vertices[3]->pos + (1 - edge_weights[0]) * edge_vertices[1]->pos;
            auto p1 = edge_weights[1] * edge_vertices[0]->pos + (1 - edge_weights[1]) * edge_vertices[2]->pos;
            auto center_v = subdiv_mesh->insert_vertex((p0 + p1) / 2);
            vertex_list = {center_v, edge_vertices[3], original_vertices[0], edge_vertices[0]};
            subdiv_mesh->insert_face(vertex_list);
            vertex_list = {center_v, edge_vertices[0], original_vertices[1], edge_vertices[1]};
            subdiv_mesh->insert_face(vertex_list);
            vertex_list = {center_v, edge_vertices[1], original_vertices[2], edge_vertices[2]};
            subdiv_mesh->insert_face(vertex_list);
            vertex_list = {center_v, edge_vertices[2], original_vertices[3], edge_vertices[3]};
            subdiv_mesh->insert_face(vertex_list);
        }
        else if (edge_vertices[0] != 0)
        {
            vertex_list = {original_vertices[0], edge_vertices[0], edge_vertices[2], original_vertices[3]};
            subdiv_mesh->insert_face(vertex_list);
            vertex_list = {edge_vertices[0], original_vertices[1], original_vertices[2], edge_vertices[2]};
            subdiv_mesh->insert_face(vertex_list);
        }
        else if (edge_vertices[1] != 0)
        {
            vertex_list = {original_vertices[0], original_vertices[1], edge_vertices[1], edge_vertices[3]};
            subdiv_mesh->insert_face(vertex_list);
            vertex_list = {original_vertices[2], original_vertices[3], edge_vertices[3], edge_vertices[1]};
            subdiv_mesh->insert_face(vertex_list);
        }
        else
        {
            vertex_list = {original_vertices[0], original_vertices[1], original_vertices[2], original_vertices[3]};
            subdiv_mesh->insert_face(vertex_list);
        }
    }
    subdiv_mesh->update_mesh();

    // (2) assign color to each vertex
    int invalid_color = 2;
    subdiv_mesh_vertex_color.assign(subdiv_mesh->get_num_of_vertices(), invalid_color); // 2: unassigned
    std::fill(subdiv_mesh_vertex_color.begin() + quad_mesh->get_num_of_vertices(), subdiv_mesh_vertex_color.end(), 0);
    int count = 0;
    for (size_t i = 0; i < corner_tag.size(); i++)
    {
        if (corner_tag[i])
        {
            subdiv_mesh_vertex_color[i] = 1;
            count++;

            auto hv = subdiv_mesh->get_vertex(i);
            hv->tag = true;
        }
    }
    for (ptrdiff_t i = quad_mesh->get_num_of_vertices(); i < subdiv_mesh->get_num_of_vertices(); i++)
    {
        auto hv = subdiv_mesh->get_vertex(i);
        hv->tag = true;
    }

    // reassign edge length via edge ring
    std::vector<bool> edge_length_tag(subdiv_mesh->get_num_of_edges(), false);
    std::vector<Real> edge_length_under_edge_ring(subdiv_mesh->get_num_of_edges(), 0);
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_edges(); i++)
    {
        auto edge = subdiv_mesh->get_edge(i);
        if (edge_length_tag[i])
            continue;

        std::vector<MeshLib::HE_edge<Real> *> edge_ring;
        edge_length_tag[i] = edge_length_tag[edge->pair->id] = true;
        edge_ring.push_back(edge);
        if (edge->face)
        {
            auto next_edge = edge->next->next;
            do
            {
                edge_length_tag[next_edge->id] = edge_length_tag[next_edge->pair->id] = true;
                edge_ring.push_back(next_edge);
                if (next_edge->pair->face)
                    next_edge = next_edge->pair->next->next;
                else
                    break;
            } while (edge_length_tag[next_edge->id] == false);
        }
        if (edge->pair->face)
        {
            auto next_edge = edge->pair->next->next;
            if (!edge_length_tag[next_edge->id])
            {
                do
                {
                    edge_length_tag[next_edge->id] = edge_length_tag[next_edge->pair->id] = true;
                    edge_ring.push_back(next_edge);
                    if (next_edge->pair->face)
                        next_edge = next_edge->pair->next->next;
                    else
                        break;
                } while (edge_length_tag[next_edge->id] == false);
            }
        }

        Real sum_len = 0;
        for (size_t j = 0; j < edge_ring.size(); j++)
        {
            sum_len += edge_ring[j]->GetLength();
        }
        Real ave_len = sum_len / edge_ring.size();
        for (size_t j = 0; j < edge_ring.size(); j++)
        {
            edge_length_under_edge_ring[edge_ring[j]->id] = ave_len;
            edge_length_under_edge_ring[edge_ring[j]->pair->id] = ave_len;
        }
    }

    ptrdiff_t nv = quad_mesh->get_num_of_vertices();
    // std::queue<MeshLib::HE_vert<Real> *> vertex_queue;
    std::vector<MeshLib::HE_vert<Real> *> path_vertices;
    std::vector<Real> path_edge_length;
    subdiv_mesh->reset_edges_tag(false);
    subdiv_mesh->reset_vertices_tag(false);

    for (size_t i = 0; i < corner_tag.size(); i++)
    {
        if (corner_tag[i] == false)
            continue;
        auto hv = subdiv_mesh->get_vertex(i);

        auto edge = hv->edge;
        do
        {

            path_vertices.resize(0);
            path_edge_length.resize(0);
            auto next_edge = edge;
            do
            {
                path_vertices.push_back(next_edge->vert);
                // path_edge_length.push_back(next_edge->GetLength());
                path_edge_length.push_back(edge_length_under_edge_ring[next_edge->id]);
                next_edge->tag = next_edge->pair->tag = true;
                if (next_edge->vert->id >= nv || corner_tag[next_edge->vert->id])
                    break;

                if (subdiv_mesh->is_on_boundary(next_edge))
                {
                    auto p_edge = next_edge->vert->edge;
                    do
                    {
                        if (subdiv_mesh->is_on_boundary(p_edge) && p_edge != next_edge->pair)
                        {
                            next_edge = p_edge;
                            break;
                        }
                        p_edge = p_edge->pair->next;
                    } while (p_edge != next_edge->vert->edge);
                }
                else
                {
                    next_edge = next_edge->next->pair->next;
                }
            } while (1);

            if (path_edge_length.size() > 1)
            {
                Real sum_len = std::accumulate(path_edge_length.begin(), path_edge_length.end(), (Real)0);
                Real accum_len = 0;
                for (size_t j = path_edge_length.size() - 1; j >= 1; j--)
                {
                    accum_len += path_edge_length[j];
                    if (!path_vertices[j - 1]->tag)
                    {
                        subdiv_mesh_vertex_color[path_vertices[j - 1]->id] = accum_len / sum_len;
                        // vertex_queue.push(path_vertices[j - 1]);
                        path_vertices[j - 1]->tag = true;
                    }
                }
            }

            edge = edge->pair->next;
        } while (edge != hv->edge);
    }

    std::queue<MeshLib::HE_edge<Real> *> edge_color_queue;
    edge_vertex_color.assign(subdiv_mesh->get_num_of_edges(), 0);
    tag_wall_edges.assign(subdiv_mesh->get_num_of_edges(), false);
    edge_color_store.clear();
    edge_color_store.push_back(std::make_pair((Real)0, (Real)0));
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_edges(); i++)
    {
        auto edge = subdiv_mesh->get_edge(i);
        if (edge > edge->pair)
            continue;
        if (subdiv_mesh_vertex_color[edge->vert->id] == 0 && subdiv_mesh_vertex_color[edge->pair->vert->id] == 0)
        {
            tag_wall_edges[edge->id] = tag_wall_edges[edge->pair->id] = true;
        }
        else
        {
            tag_wall_edges[edge->id] = tag_wall_edges[edge->pair->id] = edge->tag;
            if (edge->tag)
            {
                edge_vertex_color[edge->id] = edge_color_store.size();
                edge_vertex_color[edge->pair->id] = -(ptrdiff_t)edge_color_store.size();
                edge_color_store.push_back(std::make_pair(subdiv_mesh_vertex_color[edge->pair->vert->id], subdiv_mesh_vertex_color[edge->vert->id]));
                edge_color_queue.push(edge);
                edge_color_queue.push(edge->pair);
            }
        }
    }

    // flood edge color
    while (!edge_color_queue.empty())
    {
        auto edge = edge_color_queue.front();
        edge_color_queue.pop();
        if (edge->face == 0)
            continue;
        auto next_edge = edge->next->next;
        if (edge_vertex_color[next_edge->id] == 0)
        {
            edge_vertex_color[next_edge->id] = -edge_vertex_color[edge->id];
            if (tag_wall_edges[next_edge->id] == false)
            {
                edge_vertex_color[next_edge->pair->id] = edge_vertex_color[edge->id];
                edge_color_queue.push(next_edge->pair);
            }
        }
    }

    // while (!vertex_queue.empty())
    // {
    //     auto hv = vertex_queue.front();
    //     vertex_queue.pop();
    //     auto c = subdiv_mesh_vertex_color[hv->id];
    //     auto edge = hv->edge;
    //     do
    //     {
    //         if (edge->tag == false)
    //         {
    //             auto next_edge = edge;
    //             do
    //             {
    //                 if (next_edge->vert->id >= nv || next_edge->vert->tag == true)
    //                     break;
    //                 next_edge->tag = next_edge->pair->tag = true;
    //                 subdiv_mesh_vertex_color[next_edge->vert->id] = std::min(c, subdiv_mesh_vertex_color[next_edge->vert->id]);
    //                 next_edge = next_edge->next->pair->next;
    //             } while (1);
    //         }
    //         edge = edge->pair->next;
    //     } while (edge != hv->edge);
    // }

    // use wall edges to find clusters
    int quad_id = 0;
    quad_id_map.assign(subdiv_mesh->get_num_of_faces(), -1);
    quad2patches.resize(0);
    int num_corners = 0;
    for (size_t i = 0; i < corner_tag.size(); i++)
    {
        if (corner_tag[i] == false)
            continue;
        auto hv = subdiv_mesh->get_vertex(i);
        auto he = hv->edge;
        do
        {
            if (he->face && quad_id_map[he->face->id] == -1)
            {
                quad2patches.push_back(num_corners);
                std::queue<MeshLib::HE_face<Real> *> face_queue;
                face_queue.push(he->face);

                while (!face_queue.empty())
                {
                    auto f = face_queue.front();
                    face_queue.pop();
                    if (quad_id_map[f->id] != -1)
                        continue;
                    quad_id_map[f->id] = quad_id;
                    auto fedge = f->edge;
                    do
                    {
                        auto pair_face = fedge->pair->face;
                        if (tag_wall_edges[fedge->id] == false && pair_face && quad_id_map[pair_face->id] == -1)
                        {
                            face_queue.push(pair_face);
                        }
                        fedge = fedge->next;
                    } while (fedge != f->edge);
                }
                quad_id++;
            }
            he = he->pair->next;
        } while (he != hv->edge);
        num_corners++;
    }

    if (dualquad_mesh)
        delete dualquad_mesh;
    dualquad_mesh = new MeshLib::Mesh3D<Real>;
    dualquad_vertex_id_map.clear();
    std::vector<ptrdiff_t> inverse_vertex_id_map(subdiv_mesh->get_num_of_vertices(), -1);
    std::vector<int> wall_vertex_degree(subdiv_mesh->get_num_of_vertices(), 0);
    std::vector<bool> dual_quad_vertex_tag(subdiv_mesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_edges(); i++)
    {
        auto edge = subdiv_mesh->get_edge(i);
        if (edge > edge->pair || tag_wall_edges[edge->id] == false)
            continue;
        wall_vertex_degree[edge->vert->id]++, wall_vertex_degree[edge->pair->vert->id]++;
    }

    std::vector<bool> vert_tag;
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
    {
        if ((i < (ptrdiff_t)corner_tag.size() && corner_tag[i]) || wall_vertex_degree[i] > 2)
        {
            auto vert = dualquad_mesh->insert_vertex(subdiv_mesh->get_vertex(i)->pos);
            dualquad_vertex_id_map.push_back(i);
            inverse_vertex_id_map[i] = vert->id;
            dual_quad_vertex_tag[i] = true;

            if (i < (ptrdiff_t)corner_tag.size() && corner_tag[i])
            {
                vert_tag.push_back(true);
            }
            else
            {
                vert_tag.push_back(false);
            }
        }
    }

    subdiv_mesh->reset_edges_tag(false);
    std::vector<MeshLib::HE_vert<Real> *> facelist;
    facelist.reserve(4);
    for (size_t i = 0; i < corner_tag.size(); i++)
    {
        if (corner_tag[i] == false)
            continue;
        auto hv = subdiv_mesh->get_vertex(i);
        auto he = hv->edge;

        do
        {
            if (he->tag == false && he->face)
            {
                facelist.resize(0);
                facelist.push_back(dualquad_mesh->get_vertex(inverse_vertex_id_map[i]));
                auto cur_edge = he;
                while (facelist.size() < 4)
                {
                    cur_edge->tag = true;
                    if (dual_quad_vertex_tag[cur_edge->vert->id])
                    {
                        facelist.push_back(dualquad_mesh->get_vertex(inverse_vertex_id_map[cur_edge->vert->id]));
                        cur_edge = cur_edge->next;
                    }
                    else
                    {
                        cur_edge = cur_edge->next->pair->next;
                    }
                }

                dualquad_mesh->insert_face(facelist);
            }
            he = he->pair->next;
        } while (he != hv->edge);
    }
    dualquad_mesh->update_mesh();

    for (ptrdiff_t i = 0; i < dualquad_mesh->get_num_of_vertices(); i++)
    {
        auto hv = dualquad_mesh->get_vertex(i);
        hv->tag = vert_tag[i];
    }
    if (debug_mode)
    {
        save_mesh(subdiv_mesh, "subdiv_mesh.ply");
        SavePLYmesh_with_float_storage(dualquad_mesh, "dualquad_mesh.ply");

        // save_complex_edges("wall_edges.obj", subdiv_mesh, tag_wall_edges);
        SaveMarkedEdge_as_ply(subdiv_mesh, "wall_edges.ply", tag_wall_edges, 0.001f);

        SavePLYmesh_with_float_storage(subdiv_mesh, "subdiv_mesh_quad.ply", &quad_id_map);

        std::vector<ptrdiff_t> patch_id_map(quad_id_map.size());
        for (size_t i = 0; i < patch_id_map.size(); i++)
        {
            patch_id_map[i] = quad2patches[quad_id_map[i]];
        }
        SavePLYmesh_with_float_storage(subdiv_mesh, "subdiv_mesh_checker.ply", &patch_id_map);

        std::vector<Real> face_colors_0(subdiv_mesh->get_num_of_faces()), face_colors_1(subdiv_mesh->get_num_of_faces());
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
        {
            auto face = subdiv_mesh->get_face(i);
            auto edge = face->edge;
            auto c0 = edge_color_store[abs(edge_vertex_color[edge->id])];
            auto c1 = edge_color_store[abs(edge_vertex_color[edge->next->id])];
            face_colors_0[i] = std::min(c0.first + c0.second, c1.first + c1.second) / 2;
            face_colors_1[i] = std::min(1 - (c0.first + c0.second) / 2, 1 - (c1.first + c1.second) / 2);
        }
        SavePLYmesh_with_float_storage_and_gray_color(subdiv_mesh, "subdiv_mesh_dcdf.ply", &face_colors_0);
        SavePLYmesh_with_float_storage_and_gray_color(subdiv_mesh, "subdiv_mesh_cdf.ply", &face_colors_1);

        std::vector<TinyVector<Real, 3>> vertices;
        for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
        {
            vertices.push_back(subdiv_mesh->get_vertex(i)->pos);
        }
        SavePtsPLY("vertex_color.ply", vertices, (std::vector<TinyVector<Real, 3>> *)0, &subdiv_mesh_vertex_color);
    }
}
////////////////////////////////////////////////
// template <typename Real>
// void PatchSample<Real>::save_complex_edges(const std::string &obj_filename, MeshLib::Mesh3D<Real> *mesh, const std::vector<bool> &complex_edge_tag)
// {
//     std::ofstream mout(obj_filename);

//     std::vector<ptrdiff_t> vertex_id_map(mesh->get_num_of_vertices(), -1);
//     std::vector<bool> vertex_tag(mesh->get_num_of_vertices(), false);
//     for (ptrdiff_t i = 0; i < mesh->get_num_of_edges(); i++)
//     {
//         auto edge = mesh->get_edge(i);
//         if (edge > edge->pair || complex_edge_tag[edge->id] == false)
//             continue;
//         vertex_tag[edge->vert->id] = vertex_tag[edge->pair->vert->id] = true;
//     }

//     ptrdiff_t vcount = 1;
//     for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
//     {
//         auto vert = mesh->get_vertex(i);
//         if (vertex_tag[vert->id] == false)
//             continue;
//         mout << "v " << vert->pos[0] << " " << vert->pos[1] << " " << vert->pos[2] << std::endl;
//         vertex_id_map[vert->id] = vcount++;
//     }

//     for (ptrdiff_t i = 0; i < mesh->get_num_of_edges(); i++)
//     {
//         auto edge = mesh->get_edge(i);
//         if (edge > edge->pair || complex_edge_tag[edge->id] == false)
//             continue;
//         auto v0 = edge->vert, v1 = edge->pair->vert;
//         mout << "l " << vertex_id_map[edge->vert->id] << " " << vertex_id_map[edge->pair->vert->id] << std::endl;
//     }
//     mout.close();
// }
////////////////////////////////////////////////
template <typename Real>
void PatchSample<Real>::decompose_subdiv_mesh()
{
    subdiv_mesh->reset_faces_tag(false);
    std::queue<MeshLib::HE_face<Real> *> face_queue;
    std::vector<std::pair<ptrdiff_t, ptrdiff_t>> edge_info(subdiv_mesh->get_num_of_faces());

    split_flip.assign(subdiv_mesh->get_num_of_faces(), false);
    for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
    {
        auto face = subdiv_mesh->get_face(i);
        auto edge = face->edge;
        auto qv0 = edge->pair->vert, qv1 = edge->vert, qv2 = edge->next->vert, qv3 = edge->next->next->vert;
        auto id0 = qv0->id, id1 = qv1->id, id2 = qv2->id, id3 = qv3->id;
        if (subdiv_mesh_vertex_color[id0] == 1 || subdiv_mesh_vertex_color[id2] == 1)
        {
            face->tag = true;
            face_queue.push(face);
            edge_info[face->id] = std::make_pair(id0, id2);
        }
        else if (subdiv_mesh_vertex_color[id1] == 1 || subdiv_mesh_vertex_color[id3] == 1)
        {
            face->tag = true;
            face_queue.push(face);
            edge_info[face->id] = std::make_pair(id1, id3);
            split_flip[face->id] = true;
        }
    }

    // if (debug_mode)
    // {
    //     std::vector<TinyVector<Real, 2>> uv_coords(subdiv_mesh->get_num_of_vertices(), TinyVector<Real, 2>(2, 2));
    //     std::ofstream mout("uvobj.obj");
    //     for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
    //     {
    //         auto hv = subdiv_mesh->get_vertex(i);
    //         mout << "v " << hv->pos << std::endl;
    //     }
    //     Real vcolor_start[4], vcolor_end[4];
    //     ptrdiff_t vid[4];
    //     for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
    //     {
    //         auto face = subdiv_mesh->get_face(i);
    //         auto edge = !split_flip[i] ? face->edge : face->edge->next;
    //         auto stop_edge = !split_flip[i] ? face->edge : face->edge->next;

    //         int count = 0;
    //         do
    //         {
    //             auto store_id = edge_vertex_color[edge->id];
    //             if (store_id > 0)
    //             {
    //                 vcolor_start[count] = edge_color_store[store_id].first;
    //                 vcolor_end[count] = edge_color_store[store_id].second;
    //             }
    //             else
    //             {
    //                 vcolor_start[count] = edge_color_store[-store_id].second;
    //                 vcolor_end[count] = edge_color_store[-store_id].first;
    //             }
    //             vid[count] = edge->pair->vert->id;
    //             count++;
    //             edge = edge->next;
    //         } while (edge != stop_edge);

    //         uv_coords[vid[0]][0] = std::min(1 - vcolor_start[0], 1 - vcolor_start[1]);
    //         uv_coords[vid[0]][1] = std::min(vcolor_start[0], vcolor_start[1]);
    //         uv_coords[vid[1]][0] = std::min(1 - vcolor_end[0], 1 - vcolor_start[1]);
    //         uv_coords[vid[1]][1] = std::min(vcolor_end[0], vcolor_start[1]);
    //         uv_coords[vid[2]][0] = std::min(1 - vcolor_start[2], 1 - vcolor_start[3]);
    //         uv_coords[vid[2]][1] = std::min(vcolor_start[2], vcolor_start[3]);
    //         uv_coords[vid[3]][0] = std::min(1 - vcolor_end[2], 1 - vcolor_start[3]);
    //         uv_coords[vid[3]][1] = std::min(vcolor_end[2], vcolor_start[3]);
    //     }
    //     for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_vertices(); i++)
    //     {
    //         mout << "vt " << 1 - uv_coords[i][0] << " " << uv_coords[i][1] << "\n";
    //     }
    //     for (ptrdiff_t i = 0; i < subdiv_mesh->get_num_of_faces(); i++)
    //     {
    //         mout << "f ";
    //         auto face = subdiv_mesh->get_face(i);
    //         auto edge = face->edge;
    //         do
    //         {
    //             mout << " " << edge->pair->vert->id + 1 << "/" << edge->pair->vert->id + 1;
    //             edge = edge->next;
    //         } while (edge != face->edge);
    //         mout << "\n";
    //     }
    //     mout.close();
    // }

    while (!face_queue.empty())
    {
        auto face = face_queue.front();
        face_queue.pop();
        const auto &dv0 = edge_info[face->id].first, dv1 = edge_info[face->id].second;
        auto edge = face->edge;
        do
        {
            auto pair_face = edge->pair->face;
            if (pair_face && pair_face->tag == false && tag_wall_edges[edge->id] == false)
            {
                auto pedge = pair_face->edge;
                auto qv0 = pedge->pair->vert, qv1 = pedge->vert, qv2 = pedge->next->vert, qv3 = pedge->next->next->vert;
                auto id0 = qv0->id, id1 = qv1->id, id2 = qv2->id, id3 = qv3->id;
                if (id0 != dv0 && id0 != dv1 && id2 != dv0 && id2 != dv1)
                {
                    edge_info[pair_face->id] = std::make_pair(id0, id2);
                }
                else
                {
                    edge_info[pair_face->id] = std::make_pair(id1, id3);
                    split_flip[pair_face->id] = true;
                }

                face_queue.push(pair_face);
                pair_face->tag = true;
            }
            edge = edge->next;
        } while (edge != face->edge);
    }
}
////////////////////////////////////////////////
template <typename Real>
void PatchSample<Real>::get_grading_color_all(const int findex, const TinyVector<Real, 3> &sample_point,
                                              Real &color0, Real &color1, TinyVector<Real, 3> *color_gradient0, TinyVector<Real, 3> *color_gradient1)
{
    auto quad_face = subdiv_mesh->get_face(findex / 2);
    int tri_index = findex % 2;
    auto edge = quad_face->edge;

    MeshLib::HE_edge<Real> *myedge[2];
    if (!split_flip[findex / 2])
    {
        if (tri_index == 0)
        {
            myedge[0] = edge, myedge[1] = edge->next;
        }
        else
        {
            myedge[0] = edge->next->next, myedge[1] = edge->next->next->next;
        }
    }
    else
    {
        if (tri_index == 0)
        {
            myedge[0] = edge->next->next->next, myedge[1] = edge;
        }
        else
        {
            myedge[0] = edge->next, myedge[1] = edge->next->next;
        }
    }

    auto face_normal = (myedge[0]->vert->pos - myedge[0]->pair->vert->pos).UnitCross(myedge[1]->vert->pos - myedge[1]->pair->vert->pos);
    Real t[2], color[2];

    TinyVector<Real, 3> numerator, grad[2];
    int min_color_index = 0;
    for (int i = 0; i < 2; i++)
    {
        auto edge = myedge[i];
        auto next_edge = myedge[(i + 1) % 2];

        auto cur_edge_dir = edge->vert->pos - edge->pair->vert->pos;
        auto next_edge_dir = next_edge->vert->pos - next_edge->pair->vert->pos;

        auto denom = cur_edge_dir.Cross(next_edge_dir);
        auto numerator = (sample_point - edge->pair->vert->pos).Cross(next_edge_dir);

        int index = 0;
        Real max_denom = fabs(denom[0]);
        for (int k = 1; k < 3; k++)
        {
            if (fabs(denom[k]) > max_denom)
            {
                max_denom = fabs(denom[k]);
                index = k;
            }
        }
        t[i] = numerator[index] / denom[index];
        auto store_id = edge_vertex_color[edge->id];
        auto start_color = store_id > 0 ? edge_color_store[store_id].first : edge_color_store[-store_id].second;
        auto end_color = store_id > 0 ? edge_color_store[store_id].second : edge_color_store[-store_id].first;
        Real dc = end_color - start_color;
        color[i] = start_color + t[i] * dc;
        if (color_gradient0 || color_gradient1)
        {
            if (index == 0)
            {
                grad[i][1] = next_edge_dir[2], grad[i][2] = -next_edge_dir[1];
            }
            else if (index == 1)
            {
                grad[i][0] = -next_edge_dir[2], grad[i][2] = next_edge_dir[0];
            }
            else
            {
                grad[i][0] = next_edge_dir[1], grad[i][1] = -next_edge_dir[0];
            }
            grad[i] = (dc / denom[index]) * grad[i];
            grad[i] = grad[i] - grad[i].Dot(face_normal) * face_normal;
        }
        if (i == 0)
        {
            color0 = std::max((Real)0, std::min((Real)1, color[0]));
            if (color_gradient0)
                *color_gradient0 = grad[0];
        }
        else
        {
            color1 = std::max((Real)0, std::min((Real)1, color[1]));
            if (color_gradient1)
                *color_gradient1 = grad[1];
        }
    }
}
////////////////////////////////////////////////
template <typename Real>
Real PatchSample<Real>::get_grading_color(const int findex, const TinyVector<Real, 3> &sample_point, TinyVector<Real, 3> *color_gradient)
{
    auto quad_face = subdiv_mesh->get_face(findex / 2);
    int tri_index = findex % 2;
    auto edge = quad_face->edge;

    MeshLib::HE_edge<Real> *myedge[2];
    if (!split_flip[findex / 2])
    {
        if (tri_index == 0)
        {
            myedge[0] = edge, myedge[1] = edge->next;
        }
        else
        {
            myedge[0] = edge->next->next, myedge[1] = edge->next->next->next;
        }
    }
    else
    {
        if (tri_index == 0)
        {
            myedge[0] = edge->next->next->next, myedge[1] = edge;
        }
        else
        {
            myedge[0] = edge->next, myedge[1] = edge->next->next;
        }
    }

    auto face_normal = (myedge[0]->vert->pos - myedge[0]->pair->vert->pos).UnitCross(myedge[1]->vert->pos - myedge[1]->pair->vert->pos);
    Real t[2], color[2];

    // TinyVector<Real, 3> U, V;
    // GenerateComplementBasis(U, V, face_normal);

    TinyVector<Real, 3> numerator, grad[2];
    int min_color_index = 0;
    for (int i = 0; i < 2; i++)
    {
        auto edge = myedge[i];
        auto next_edge = myedge[(i + 1) % 2];

        auto cur_edge_dir = edge->vert->pos - edge->pair->vert->pos;
        auto next_edge_dir = next_edge->vert->pos - next_edge->pair->vert->pos;

        auto denom = cur_edge_dir.Cross(next_edge_dir);
        auto numerator = (sample_point - edge->pair->vert->pos).Cross(next_edge_dir);

        int index = 0;
        Real max_denom = fabs(denom[0]);
        for (int k = 1; k < 3; k++)
        {
            if (fabs(denom[k]) > max_denom)
            {
                max_denom = fabs(denom[k]);
                index = k;
            }
        }
        t[i] = numerator[index] / denom[index];
        auto store_id = edge_vertex_color[edge->id];
        auto start_color = store_id > 0 ? edge_color_store[store_id].first : edge_color_store[-store_id].second;
        auto end_color = store_id > 0 ? edge_color_store[store_id].second : edge_color_store[-store_id].first;
        Real dc = end_color - start_color;
        color[i] = start_color + t[i] * dc;
        if (color_gradient)
        {
            if (index == 0)
            {
                grad[i][1] = next_edge_dir[2], grad[i][2] = -next_edge_dir[1];
            }
            else if (index == 1)
            {
                grad[i][0] = -next_edge_dir[2], grad[i][2] = next_edge_dir[0];
            }
            else
            {
                grad[i][0] = next_edge_dir[1], grad[i][1] = -next_edge_dir[0];
            }
            grad[i] = (dc / denom[index]) * grad[i];
            grad[i] = grad[i] - grad[i].Dot(face_normal) * face_normal;
            // grad[i].Normalize();

            // TinyVector<Real, 2> cur_edge_dir2d(cur_edge_dir.Dot(U), cur_edge_dir.Dot(V));
            // TinyVector<Real, 2> sample_point_2d((sample_point - edge->pair->vert->pos).Dot(U), (sample_point - edge->pair->vert->pos).Dot(V));
            // TinyVector<Real, 2> next_edge_dir2d(next_edge_dir.Dot(U), next_edge_dir.Dot(V));
            // auto denom2d = cur_edge_dir2d.Cross(next_edge_dir2d)[0];
            // auto numerator2d = sample_point_2d.Cross(next_edge_dir2d)[0];
            // if (fabs(t[i] - numerator2d / denom2d) > 1.0e-6)
            // {
            //     std::cout << t[i] << " " << numerator2d / denom2d << std::endl;
            // }
            // auto new_grad = (dc / denom2d) * (next_edge_dir2d[1] * U - next_edge_dir2d[0] * V);
            // if ((new_grad - grad[i]).Length() > 1.0e-6)
            // {
            //     std::cout << "grad error: " << new_grad << ' ' << grad[i] << std::endl;
            // }
        }
        if (i == 1 && color[1] < color[0])
            min_color_index = 1;
    }
    if (color_gradient)
        *color_gradient = grad[min_color_index];

    return std::max((Real)0, std::min((Real)1, color[min_color_index]));
}

////////////////////////////////////////////////
template <typename Real>
void PatchSample<Real>::set_mesh_as_complex(MeshLib::Mesh3D<Real> *mesh,
                                            std::vector<bool> &complex_edge_tag,
                                            std::vector<bool> &corner_tag,
                                            std::vector<std::vector<MeshLib::HE_edge<Real> *>> &complex_edge_loops,
                                            std::vector<std::vector<ptrdiff_t>> &complex_edge_loops_corner_starting_edges,
                                            std::vector<std::vector<ptrdiff_t>> &complex_edge_loops_neighbor_cluster_ids,
                                            std::vector<ptrdiff_t> &complex_edge_loops_cluster_ids,
                                            std::unordered_map<complex_arc, complex_arc_info> &complex_arcs,
                                            size_t &arc_group_num)
{
    mesh->reset_edges_tag(false);
    complex_edge_tag.assign(mesh->get_num_of_edges(), true);
    corner_tag.assign(mesh->get_num_of_vertices(), true);
    complex_edge_loops.resize(0);
    complex_edge_loops_corner_starting_edges.resize(0);
    complex_edge_loops_neighbor_cluster_ids.resize(0);
    complex_edge_loops_cluster_ids.resize(0);
    for (size_t i = 0; i < complex_edge_tag.size(); i++)
    {
        auto edge = mesh->get_edge(i);
        if (complex_edge_tag[i] && edge->tag == false)
        {
            if (!edge->face)
                continue;

            complex_edge_loops_cluster_ids.emplace_back(edge->face->id);
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
                        throw std::runtime_error("Error: more than four corners on one complex edge loop! <PatchSample.cpp:set_mesh_as_complex>");
                    }
                    else
                    {
                        neighbor_cluster_ids.emplace_back(edge->pair->face ? edge->pair->face->id : -1);
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

            complex_arc arc(start_vert->id, end_vert->id, start_edge->face->id, start_edge->pair->face ? start_edge->pair->face->id : -1);
            auto iter = complex_arcs.find(arc);
            if (iter != complex_arcs.end())
            {
                auto nedge = boundary_edge_loop[corner_indices_on_loop[(j + 2) % 4]];
                iter->second.add_ring_neighbor(corner_verts[(j + 2) % 4]->id, corner_verts[(j + 3) % 4]->id,
                                               nedge->face->id, nedge->pair->face ? nedge->pair->face->id : -1);
            }
            else
            {
                auto nedge = boundary_edge_loop[corner_indices_on_loop[(j + 2) % 4]];
                complex_arc_info arc_info;
                arc_info.add_ring_neighbor(corner_verts[(j + 2) % 4]->id, corner_verts[(j + 3) % 4]->id,
                                           nedge->face->id, nedge->pair->face ? nedge->pair->face->id : -1);
                arc_info.arclength = 0;
                for (size_t k = 0; k < boundary_edge_loop.size(); k++)
                {
                    auto edge = boundary_edge_loop[(k + corner_indices_on_loop[j]) % boundary_edge_loop.size()];
                    if (edge == end_edge)
                        break;
                    arc_info.arclength += edge->GetLength();
                }
                complex_arcs[arc] = arc_info;
            }
        }
    }

    arc_group_num = 0;
    for (auto &c_arc : complex_arcs)
    {
        if (c_arc.second.visited)
            continue;
        std::queue<complex_arc> arc_queue;
        arc_queue.push(c_arc.first);
        c_arc.second.orientation = true;

        while (!arc_queue.empty())
        {
            auto arc = arc_queue.front();
            arc_queue.pop();
            auto iter = complex_arcs.find(arc);
            if (iter->second.visited)
                continue;

            bool orientation = iter->second.orientation;
            iter->second.visited = true;
            iter->second.group_id = (int)arc_group_num;

            if (iter->second.ring_neighbor_arc_1[0] != -1 && iter->second.ring_neighbor_arc_1[1] != -1)
            {
                complex_arc arc1(iter->second.ring_neighbor_arc_1[0], iter->second.ring_neighbor_arc_1[1],
                                 iter->second.cluster_id_1[0], iter->second.cluster_id_1[1]);
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
                        throw std::runtime_error("Error: cannot determine the orientation of arcs! <patchsample.cpp:set_mesh_as_complex>");
                    }
                }
            }

            if (iter->second.ring_neighbor_arc_2[0] != -1 && iter->second.ring_neighbor_arc_2[1] != -1)
            {
                complex_arc arc2(iter->second.ring_neighbor_arc_2[0], iter->second.ring_neighbor_arc_2[1],
                                 iter->second.cluster_id_2[0], iter->second.cluster_id_2[1]);
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
                        throw std::runtime_error("Error: cannot determine the orientation of arcs! <patchsample.cpp:set_mesh_as_complex>");
                    }
                }
            }
        }
        arc_group_num++;
    }

    complex_edge_tag.assign(mesh->get_num_of_edges(), true);
}
////////////////////////////////////////////////
template class PatchSample<double>;