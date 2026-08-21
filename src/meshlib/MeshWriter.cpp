#include "MeshWriter.h"
#include "happly.h"
#include "tinycolormap.hpp"
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <random>
#include <chrono>
#include "GraphColoring.h"
#include "Libboard/Board.h"
#include "tiny_gltf.h"
//////////////////////////////////////////////////////////////////////////
std::array<unsigned char, 3> hsvToRgb(float h, float s, float v)
{
    const float c = v * s;
    const float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    const float m = v - c;

    float r = 0, g = 0, b = 0;

    if (h < 1.0f / 6.0f)
    {
        r = c;
        g = x;
        b = 0.0f;
    }
    else if (h < 2.0f / 6.0f)
    {
        r = x;
        g = c;
        b = 0.0f;
    }
    else if (h < 3.0f / 6.0f)
    {
        r = 0.0f;
        g = c;
        b = x;
    }
    else if (h < 4.0f / 6.0f)
    {
        r = 0.0f;
        g = x;
        b = c;
    }
    else if (h < 5.0f / 6.0f)
    {
        r = x;
        g = 0.0f;
        b = c;
    }
    else
    {
        r = c;
        g = 0.0f;
        b = x;
    }

    return std::array<unsigned char, 3>({static_cast<unsigned char>((r + m) * 255.0f),
                                         static_cast<unsigned char>((g + m) * 255.0f),
                                         static_cast<unsigned char>((b + m) * 255.0f)});
}
std::vector<std::array<unsigned char, 3>> generateDistinctColors(int n)
{
    std::vector<std::array<unsigned char, 3>> colors;
    colors.reserve(n);

    constexpr float golden_ratio = 0.618033988749f;
    // thread_local std::mt19937 gen((unsigned int)std::chrono::steady_clock::now().time_since_epoch().count());
    thread_local std::mt19937 gen(0);
    std::uniform_real_distribution<float> sat_dist(0.6f, 1.0f);
    std::uniform_real_distribution<float> val_dist(0.7f, 1.0f);
    for (int i = 0; i < n; ++i)
    {
        const float hue = std::fmod(i * golden_ratio, 1.0f);
        const float saturation = sat_dist(gen);
        const float brightness = val_dist(gen);
        colors.push_back(hsvToRgb(hue, saturation, brightness));
    }
    return colors;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename, const std::vector<ptrdiff_t> *face_cluster_ids, bool use_coloring_algorithm)
{
    if (!m_pmesh)
        return;

    std::vector<float> vertexX, vertexY, vertexZ;
    vertexX.resize(m_pmesh->get_num_of_vertices());
    vertexY.resize(m_pmesh->get_num_of_vertices());
    vertexZ.resize(m_pmesh->get_num_of_vertices());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto v = m_pmesh->get_vertex(i);
        vertexX[i] = static_cast<float>(v->pos[0]), vertexY[i] = static_cast<float>(v->pos[1]), vertexZ[i] = static_cast<float>(v->pos[2]);
    }
    std::vector<std::vector<size_t>> faceIndices;
    faceIndices.reserve(m_pmesh->get_num_of_faces());
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto f = m_pmesh->get_face(i);
        auto he = f->edge;
        std::vector<size_t> vertex_list;
        vertex_list.reserve(f->valence);
        do
        {
            vertex_list.emplace_back(static_cast<size_t>(he->vert->id));
            he = he->next;
        } while (he != f->edge);
        faceIndices.emplace_back(vertex_list);
    }

    std::vector<std::array<unsigned char, 3>> ply_face_color;
    if (face_cluster_ids)
    {
        if (!use_coloring_algorithm)
        {
            auto max_cluster_id = *std::max_element(face_cluster_ids->begin(), face_cluster_ids->end());
            auto cluster_colors = generateDistinctColors(static_cast<int>(max_cluster_id) + 1);
            ply_face_color.reserve(m_pmesh->get_num_of_faces());
            for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
            {
                ply_face_color.emplace_back(cluster_colors[(*face_cluster_ids)[i]]);
            }
        }
        else
        {

            auto num_complex = *std::max_element(face_cluster_ids->begin(), face_cluster_ids->end()) + 1;
            std::vector<std::unordered_set<size_t>> graph_edges(num_complex);

            for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
            {
                auto hv = m_pmesh->get_vertex(i);
                auto he = hv->edge;
                std::unordered_set<size_t> clist;
                do
                {
                    if (he->face)
                        clist.insert((*face_cluster_ids)[he->face->id]);
                    he = he->pair->next;
                } while (hv->edge != he);
                std::vector<size_t> clist_vec(clist.begin(), clist.end());
                for (size_t j = 0; j < clist_vec.size(); j++)
                {
                    for (size_t k = j + 1; k < clist_vec.size(); k++)
                    {
                        graph_edges[clist_vec[j]].insert(clist_vec[k]);
                        graph_edges[clist_vec[k]].insert(clist_vec[j]);
                    }
                }
            }
            std::vector<std::vector<size_t>> colored_vertices;
            greedy_graph_coloring(num_complex, graph_edges, colored_vertices);

            std::vector<size_t> complex_color(face_cluster_ids->size());
            for (size_t i = 0; i < colored_vertices.size(); i++)
            {
                for (size_t j = 0; j < colored_vertices[i].size(); j++)
                {
                    complex_color[colored_vertices[i][j]] = i;
                }
            }
            auto cluster_colors = generateDistinctColors(static_cast<int>(colored_vertices.size()));

            ply_face_color.reserve(m_pmesh->get_num_of_faces());
            for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
            {
                ply_face_color.emplace_back(cluster_colors[complex_color[(*face_cluster_ids)[i]]]);
            }
        }
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addFaceIndices(faceIndices);
    if (!ply_face_color.empty())
        plyOut.addFaceColors(ply_face_color);

    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveChartEdge_as_ply(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename, const std::vector<ptrdiff_t> &face_cluster_ids, float normaloffset)
{
    if (!m_pmesh)
        return;
    std::vector<bool> marked_edge_tag(m_pmesh->get_num_of_edges(), false);
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge < edge->pair && (m_pmesh->is_on_boundary(edge) || (face_cluster_ids[edge->face->id] != face_cluster_ids[edge->pair->face->id])))
            marked_edge_tag[edge->id] = marked_edge_tag[edge->pair->id] = true;
    }
    SaveMarkedEdge_as_ply(m_pmesh, filename, marked_edge_tag, normaloffset);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveMarkedEdge_as_ply(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename, const std::vector<bool> &marked_edge_tag, float normaloffset)
{
    if (!m_pmesh)
        return;
    std::vector<ptrdiff_t> vertex_id_map(m_pmesh->get_num_of_vertices(), -1);
    std::vector<bool> vertex_tag(m_pmesh->get_num_of_vertices(), false);
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge > edge->pair || marked_edge_tag[edge->id] == false)
            continue;
        vertex_tag[edge->vert->id] = vertex_tag[edge->pair->vert->id] = true;
    }
    std::vector<float> vertexX, vertexY, vertexZ;
    ptrdiff_t vcount = 0;
    vertexX.reserve(m_pmesh->get_num_of_vertices());
    vertexY.reserve(m_pmesh->get_num_of_vertices());
    vertexZ.reserve(m_pmesh->get_num_of_vertices());
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto vert = m_pmesh->get_vertex(i);
        if (vertex_tag[vert->id] == false)
            continue;
        vertexX.emplace_back(static_cast<float>(vert->pos[0]) + normaloffset * static_cast<float>(vert->normal[0]));
        vertexY.emplace_back(static_cast<float>(vert->pos[1]) + normaloffset * static_cast<float>(vert->normal[1]));
        vertexZ.emplace_back(static_cast<float>(vert->pos[2]) + normaloffset * static_cast<float>(vert->normal[2]));
        vertex_id_map[vert->id] = vcount++;
    }
    std::vector<int> chart_edge_indices[2];
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge > edge->pair || marked_edge_tag[edge->id] == false)
            continue;
        auto v0 = edge->vert, v1 = edge->pair->vert;
        chart_edge_indices[0].emplace_back(static_cast<int>(vertex_id_map[v0->id]));
        chart_edge_indices[1].emplace_back(static_cast<int>(vertex_id_map[v1->id]));
    }
    happly::PLYData plyOut;
    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addElement("edge", chart_edge_indices[0].size());
    plyOut.getElement("edge").addProperty<int>("vertex1", chart_edge_indices[0]);
    plyOut.getElement("edge").addProperty<int>("vertex2", chart_edge_indices[1]);
    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveEdge_as_ply(const std::string &filename, const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::pair<size_t, size_t>> &edges)
{
    std::vector<float> vertexX(vertices.size()), vertexY(vertices.size()), vertexZ(vertices.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(vertices.size()); i++)
    {
        vertexX[i] = static_cast<float>(vertices[i][0]);
        vertexY[i] = static_cast<float>(vertices[i][1]);
        vertexZ[i] = static_cast<float>(vertices[i][2]);
    }
    std::vector<int> chart_edge_indices[2];
    chart_edge_indices[0].resize(edges.size()), chart_edge_indices[1].resize(edges.size());
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(edges.size()); i++)
    {
        chart_edge_indices[0][i] = static_cast<int>(edges[i].first);
        chart_edge_indices[1][i] = static_cast<int>(edges[i].second);
    }
    happly::PLYData plyOut;
    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addElement("edge", chart_edge_indices[0].size());
    plyOut.getElement("edge").addProperty<int>("vertex1", chart_edge_indices[0]);
    plyOut.getElement("edge").addProperty<int>("vertex2", chart_edge_indices[1]);
    plyOut.write(filename, happly::DataFormat::Binary);
}
template <typename Real>
void SaveEdge_as_ply(const std::string &filename, const std::vector<MeshLib::HE_edge<Real> *> &edges)
{
    std::vector<float> vertexX(2 * edges.size()), vertexY(2 * edges.size()), vertexZ(2 * edges.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(edges.size()); i++)
    {
        vertexX[2 * i] = static_cast<float>(edges[i]->pair->vert->pos[0]);
        vertexY[2 * i] = static_cast<float>(edges[i]->pair->vert->pos[1]);
        vertexZ[2 * i] = static_cast<float>(edges[i]->pair->vert->pos[2]);
        vertexX[2 * i + 1] = static_cast<float>(edges[i]->vert->pos[0]);
        vertexY[2 * i + 1] = static_cast<float>(edges[i]->vert->pos[1]);
        vertexZ[2 * i + 1] = static_cast<float>(edges[i]->vert->pos[2]);
    }
    std::vector<int> chart_edge_indices[2];
    chart_edge_indices[0].resize(edges.size()), chart_edge_indices[1].resize(edges.size());
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(edges.size()); i++)
    {
        chart_edge_indices[0][i] = static_cast<int>(2 * i);
        chart_edge_indices[1][i] = static_cast<int>(2 * i + 1);
    }
    happly::PLYData plyOut;
    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addElement("edge", chart_edge_indices[0].size());
    plyOut.getElement("edge").addProperty<int>("vertex1", chart_edge_indices[0]);
    plyOut.getElement("edge").addProperty<int>("vertex2", chart_edge_indices[1]);
    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
void SaveMeshlabMLP(const std::string &mlpfile, const std::string &meshfile, const std::string &chartedgefile,
                    int meshedge_width, int charedge_wirewidth)
{
    std::ofstream mlp(mlpfile);
    if (!mlp.is_open())
    {
        std::cerr << "Failed to open MLP file: " << mlpfile << '\n';
        return;
    }

    auto mesh_edge_string = meshedge_width > 0 ? "101001000000010000000100000001010101000010100000110010111011100000001001" : "100001000000000000000100000001010101000010100000100010111011100000001001";
    auto chart_edge_string = charedge_wirewidth > 0 ? "001000000000010000000100000001000100000010100000110100111011110000101001" : "000000000000000000000100000001000100000010100000100100111011110000101001";

    int mesh_edge_width_ = std::max(std::min(meshedge_width, 5), 1);
    int chart_edge_wirewidth_ = std::max(std::min(charedge_wirewidth, 5), 1);

    mlp << "<!DOCTYPE MeshLabDocument>\n";
    mlp << "<MeshLabProject>\n";
    mlp << " <MeshGroup>\n";
    mlp << "  <MLMesh label=\"" << "Mesh" << "\" filename=\"" << meshfile << "\" visible=\"1\" idInFile=\"-1\">\n";
    mlp << "   <MLMatrix44>\n";
    mlp << "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n";
    mlp << "   </MLMatrix44>\n";
    mlp << "   <RenderingOption wireWidth=\"" << mesh_edge_width_ << "\" pointColor=\"252 233 79 255\" pointSize=\"3\" wireColor=\"64 64 64 255\" solidColor=\"192 192 192 255\" boxColor=\"234 234 234 255\">" << mesh_edge_string << "</RenderingOption>\n";
    mlp << "  </MLMesh>\n";
    mlp << "  <MLMesh label=\"" << "ChartEdges" << "\" filename=\"" << chartedgefile << "\" visible=\"1\" idInFile=\"-1\">\n";
    mlp << "   <MLMatrix44>\n";
    mlp << "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n";
    mlp << "   </MLMatrix44>\n";
    mlp << "   <RenderingOption wireWidth=\"" << chart_edge_wirewidth_ << "\" pointColor=\"255 255 255 255\" pointSize=\"15\" wireColor=\"0 0 0 255\" solidColor=\"192 192 192 255\" boxColor=\"234 234 234 255\">001000000000010000000100000001000100000010100000110100111011110000101001</RenderingOption>\n";
    mlp << "  </MLMesh>\n";
    mlp << " </MeshGroup>\n";
    mlp << " <RasterGroup/>\n";
    mlp << "</MeshLabProject>\n";
    mlp.close();
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SaveComponents(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename)
{
    if (!m_pmesh)
        return;

    m_pmesh->reset_faces_tag(false);
    int component_count = 0;

    for (int i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto face = m_pmesh->get_face(i);
        if (face->tag)
            continue;

        std::queue<MeshLib::HE_face<Real> *> facequeue;
        facequeue.push(face);
        face->tag = true;
        std::vector<MeshLib::HE_face<Real> *> component;
        component.emplace_back(face);
        size_t num_quads = 0, num_totals = 0;
        while (!facequeue.empty())
        {
            auto f = facequeue.front();
            facequeue.pop();

            auto he = f->edge;
            do
            {
                if (he->pair->face && !he->pair->face->tag)
                {
                    facequeue.push(he->pair->face);
                    he->pair->face->tag = true;
                    component.emplace_back(he->pair->face);
                    if (he->pair->face->valence == 4)
                        num_quads++;
                    num_totals++;
                }
                he = he->next;
            } while (he != f->edge);
        }
        Real quad_ratio = num_quads / static_cast<Real>(num_totals);
        if (quad_ratio < Real(0.85) || num_quads < 20)
            continue;
        std::string plyfilename = filename;
        size_t lastindex = plyfilename.find_last_of(".");
        plyfilename = plyfilename.substr(0, lastindex);
        plyfilename += "_component_" + std::to_string(component_count) + ".ply";
        save_component_as_ply(component, plyfilename);
        component_count++;
    }
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void save_component_as_ply(const std::vector<MeshLib::HE_face<Real> *> &component, const std::string &filename)
{
    if (component.empty())
        return;

    std::unordered_map<MeshLib::HE_vert<Real> *, size_t> vert_map;
    std::vector<float> vertexX, vertexY, vertexZ;
    std::vector<std::vector<size_t>> faceIndices;
    faceIndices.reserve(component.size());
    for (auto f : component)
    {
        auto he = f->edge;
        do
        {
            if (vert_map.find(he->vert) == vert_map.end())
            {
                vert_map[he->vert] = vertexX.size();
                vertexX.emplace_back(static_cast<float>(he->vert->pos[0]));
                vertexY.emplace_back(static_cast<float>(he->vert->pos[1]));
                vertexZ.emplace_back(static_cast<float>(he->vert->pos[2]));
            }
            he = he->next;
        } while (he != f->edge);
    }

    for (auto f : component)
    {
        auto he = f->edge;
        std::vector<size_t> face;
        face.reserve(f->valence);
        do
        {
            face.emplace_back(vert_map[he->vert]);
            he = he->next;
        } while (he != f->edge);
        faceIndices.emplace_back(face);
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addFaceIndices(faceIndices);

    plyOut.write(filename.c_str(), happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void save_mesh(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, const std::string &filename)
{
    // get the file extension
    size_t lastindex = filename.find_last_of(".");
    std::string ext = filename.substr(lastindex + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                   { return std::tolower(c); }); //
    if (ext == "obj")
    {
        // Ultra-fast OBJ writing with C-style formatting and preallocated buffers
        std::ofstream objfile(filename, std::ios::out | std::ios::binary);
        objfile.tie(nullptr); // Disable tie for better performance

        constexpr size_t BUFFER_SIZE = 256 * 1024; // 256KB buffer for even better throughput
        objfile.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);

        // Preallocate a large string for vertices and faces
        std::string vertexBuffer;
        vertexBuffer.reserve(vertices.size() * 64);

        for (const auto &vertex : vertices)
        {
            char line[128];
            int len = snprintf(line, sizeof(line), "v %.16g %.16g %.16g\n",
                               static_cast<double>(vertex[0]), static_cast<double>(vertex[1]), static_cast<double>(vertex[2]));
            vertexBuffer.append(line, len);
        }
        objfile.write(vertexBuffer.data(), vertexBuffer.size());

        std::string faceBuffer;
        faceBuffer.reserve(face_indices.size() * 32);

        for (const auto &face : face_indices)
        {
            faceBuffer.append("f");
            for (const auto &idx : face)
            {
                char idxStr[16];
                int idxLen = snprintf(idxStr, sizeof(idxStr), " %zu", idx + 1);
                faceBuffer.append(idxStr, idxLen);
            }
            faceBuffer.append("\n");
        }
        objfile.write(faceBuffer.data(), faceBuffer.size());

        objfile.close();
    }
    else if (ext == "off")
    {
        // Use large buffer and C-style formatting for efficient OFF writing
        std::ofstream offfile(filename);
        offfile.tie(nullptr); // Untie for performance

        constexpr size_t BUFFER_SIZE = 64 * 1024; // 64KB buffer
        offfile.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);

        offfile.precision(16);
        offfile << "OFF\n";
        offfile << vertices.size() << " " << face_indices.size() << " 0\n";

        // Write vertices efficiently
        for (const auto &vertex : vertices)
        {
            char line[128];
            int len = snprintf(line, sizeof(line), "%.16g %.16g %.16g\n",
                               static_cast<double>(vertex[0]), static_cast<double>(vertex[1]), static_cast<double>(vertex[2]));
            offfile.write(line, len);
        }

        // Write faces efficiently
        for (const auto &face : face_indices)
        {
            char line[256];
            int pos = snprintf(line, sizeof(line), "%zu", face.size());
            for (const auto &idx : face)
            {
                pos += snprintf(line + pos, sizeof(line) - pos, " %zu", idx);
            }
            line[pos++] = '\n';
            offfile.write(line, pos);
        }

        offfile.close();
    }
    else if (ext == "ply")
    {
        std::vector<float> vertexX(vertices.size()), vertexY(vertices.size()), vertexZ(vertices.size());
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(vertices.size()); i++)
        {
            vertexX[i] = static_cast<float>(vertices[i][0]), vertexY[i] = static_cast<float>(vertices[i][1]), vertexZ[i] = static_cast<float>(vertices[i][2]);
        }
        happly::PLYData plyOut;

        plyOut.addElement("vertex", vertexX.size());
        plyOut.getElement("vertex").addProperty<float>("x", vertexX);
        plyOut.getElement("vertex").addProperty<float>("y", vertexY);
        plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
        plyOut.addFaceIndices(face_indices);
        plyOut.write(filename, happly::DataFormat::Binary);
    }
    else if (ext == "glb" || ext == "gltf")
    {
        // warning: only support triangular mesh
        std::vector<float> my_vertices(3 * vertices.size());
        size_t nf = 0;
        std::vector<size_t> face_start_index(face_indices.size(), 0);
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(face_indices.size()) - 1; i++)
        {
            nf += face_indices[i].size() - 2;
            face_start_index[i + 1] = nf;
        }
        nf += face_indices.back().size() - 2;
        std::vector<uint32_t> my_indices(3 * nf);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(vertices.size()); i++)
        {
            my_vertices[3 * i] = static_cast<float>(vertices[i][0]);
            my_vertices[3 * i + 1] = static_cast<float>(vertices[i][1]);
            my_vertices[3 * i + 2] = static_cast<float>(vertices[i][2]);
        }
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(face_indices.size()); i++)
        {
            auto start = face_start_index[i] * 3;
            for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(face_indices[i].size()) - 2; j++)
            {
                my_indices[start + 3 * j] = static_cast<uint32_t>(face_indices[i][0]);
                my_indices[start + 3 * j + 1] = static_cast<uint32_t>(face_indices[i][j + 1]);
                my_indices[start + 3 * j + 2] = static_cast<uint32_t>(face_indices[i][j + 2]);
            }
        }

        tinygltf::Model model;
        tinygltf::TinyGLTF gltf;

        // Buffer
        tinygltf::Buffer buffer;
        buffer.data.insert(buffer.data.end(),
                           reinterpret_cast<const unsigned char *>(my_vertices.data()),
                           reinterpret_cast<const unsigned char *>(my_vertices.data() + my_vertices.size()));
        size_t vertexBufferSize = buffer.data.size();
        buffer.data.insert(buffer.data.end(),
                           reinterpret_cast<const unsigned char *>(my_indices.data()),
                           reinterpret_cast<const unsigned char *>(my_indices.data() + my_indices.size()));
        model.buffers.push_back(buffer);

        // BufferViews
        tinygltf::BufferView posView;
        posView.buffer = 0;
        posView.byteOffset = 0;
        posView.byteLength = my_vertices.size() * sizeof(float);
        posView.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        model.bufferViews.push_back(std::move(posView));

        tinygltf::BufferView idxView;
        idxView.buffer = 0;
        idxView.byteOffset = vertexBufferSize;
        idxView.byteLength = my_indices.size() * sizeof(uint32_t);
        idxView.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
        model.bufferViews.push_back(std::move(idxView));

        // Accessors
        tinygltf::Accessor posAccessor;
        posAccessor.bufferView = 0;
        posAccessor.byteOffset = 0;
        posAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        posAccessor.count = vertices.size();
        posAccessor.type = TINYGLTF_TYPE_VEC3;
        model.accessors.push_back(std::move(posAccessor));

        tinygltf::Accessor idxAccessor;
        idxAccessor.bufferView = 1;
        idxAccessor.byteOffset = 0;
        idxAccessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
        idxAccessor.count = my_indices.size();
        idxAccessor.type = TINYGLTF_TYPE_SCALAR;
        model.accessors.push_back(std::move(idxAccessor));

        // Mesh
        tinygltf::Primitive primitive;
        primitive.attributes["POSITION"] = 0;
        primitive.indices = 1;
        primitive.mode = TINYGLTF_MODE_TRIANGLES;

        tinygltf::Mesh mesh;
        mesh.primitives.push_back(std::move(primitive));
        model.meshes.push_back(std::move(mesh));

        // Node and Scene
        tinygltf::Node node;
        node.mesh = 0;
        model.nodes.push_back(std::move(node));

        tinygltf::Scene scene;
        scene.nodes.push_back(0);
        model.scenes.push_back(scene);
        model.defaultScene = 0;

        // Write GLB
        gltf.WriteGltfSceneToFile(&model, filename, false, false, false, true);
    }
    else if (ext == "vtk")
    {
        std::ofstream vtkfile(filename);
        vtkfile << "# vtk DataFile Version 2.0\n"
                << "Polygonal Mesh\n"
                << "ASCII\n"
                << "DATASET UNSTRUCTURED_GRID\n"
                << "POINTS " << vertices.size() << " double\n";
        for (size_t i = 0; i < vertices.size(); ++i)
            vtkfile << std::scientific << vertices[i] << '\n';
        size_t total_num_indices = 0;
        for (size_t i = 0; i < face_indices.size(); ++i)
            total_num_indices += face_indices[i].size() + 1;
        vtkfile << "CELLS " << face_indices.size() << " " << total_num_indices << '\n';
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            vtkfile << face_indices[i].size();
            for (const auto j : face_indices[i])
                vtkfile << " " << j;
            vtkfile << '\n';
        }
        vtkfile << "CELL_TYPES " << face_indices.size() << '\n';
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            if (face_indices[i].size() == 3)
                vtkfile << "5\n"; // VTK_TRIANGLE
            else if (face_indices[i].size() == 4)
                vtkfile << "9\n"; // VTK_QUAD
            else
                vtkfile << "7\n"; // VTK_POLYGON
        }
        vtkfile.close();
    }
    else if (ext == "vtp")
    {
        std::ofstream vtpfile(filename);
        vtpfile << "<?xml version=\"1.0\"?>\n"
                << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                << " <PolyData>\n"
                << "  <Piece NumberOfPoints=\"" << vertices.size() << "\" NumberOfPolys=\"" << face_indices.size() << "\">\n"
                << "   <Points>\n"
                << "    <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < vertices.size(); ++i)
            vtpfile << std::scientific << vertices[i] << " ";
        vtpfile << "\n"
                << "    </DataArray>\n"
                << "   </Points>\n"
                << "   <Polys>\n"
                << "    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            for (const auto j : face_indices[i])
                vtpfile << j << " ";
        }
        vtpfile << "\n"
                << "    </DataArray>\n"
                << "    <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        size_t offset = 0;
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            offset += face_indices[i].size();
            vtpfile << offset << " ";
        }
        vtpfile << "\n"
                << "    </DataArray>\n"
                << "   </Polys>\n"
                << "  </Piece>\n"
                << " </PolyData>\n"
                << "</VTKFile>\n";
        vtpfile.close();
    }
    else if (ext == "stl")
    {
        std::ofstream bstlfile(filename, std::ios::binary);
        char header[80] = "Binary STL generated by QuadTools";
        bstlfile.write(header, 80);
        uint32_t num_triangles = 0;
        for (size_t i = 0; i < face_indices.size(); ++i)
            num_triangles += static_cast<uint32_t>(face_indices[i].size() - 2);
        bstlfile.write(reinterpret_cast<const char *>(&num_triangles), 4);
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            if (face_indices[i].size() < 3)
                continue;
            TinyVector<Real, 3> v0 = vertices[face_indices[i][0]];
            for (size_t j = 1; j < face_indices[i].size() - 1; ++j)
            {
                TinyVector<Real, 3> v1 = vertices[face_indices[i][j]];
                TinyVector<Real, 3> v2 = vertices[face_indices[i][j + 1]];
                TinyVector<Real, 3> normal = (v1 - v0).UnitCross(v2 - v0);
                float normal_f[3] = {static_cast<float>(normal[0]), static_cast<float>(normal[1]), static_cast<float>(normal[2])};
                bstlfile.write(reinterpret_cast<const char *>(normal_f), 12);
                float v0_f[3] = {static_cast<float>(v0[0]), static_cast<float>(v0[1]), static_cast<float>(v0[2])};
                float v1_f[3] = {static_cast<float>(v1[0]), static_cast<float>(v1[1]), static_cast<float>(v1[2])};
                float v2_f[3] = {static_cast<float>(v2[0]), static_cast<float>(v2[1]), static_cast<float>(v2[2])};
                bstlfile.write(reinterpret_cast<const char *>(v0_f), 12);
                bstlfile.write(reinterpret_cast<const char *>(v1_f), 12);
                bstlfile.write(reinterpret_cast<const char *>(v2_f), 12);
                uint16_t attribute_byte_count = 0;
                bstlfile.write(reinterpret_cast<const char *>(&attribute_byte_count), 2);
            }
        }
        bstlfile.close();
    }
    else if (ext == "usd")
    {
        std::ofstream usdfile(filename);
        usdfile << "#usda 1.0\n"
                << "(\n"
                << "    defaultPrim = \"Mesh\"\n"
                << ")\n"
                << "\n"
                << "def Mesh \"Mesh\"\n"
                << "{\n";
        usdfile << "    int[] faceVertexCounts = [";
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            usdfile << face_indices[i].size();
            if (i != face_indices.size() - 1)
                usdfile << ", ";
        }
        usdfile << "]\n";
        usdfile << "    int[] faceVertexIndices = [";
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            for (size_t j = 0; j < face_indices[i].size(); ++j)
            {
                usdfile << face_indices[i][j];
                if (i != face_indices.size() - 1 || j != face_indices[i].size() - 1)
                    usdfile << ", ";
            }
        }
        usdfile << "]\n";
        usdfile << "    point3f[] points = [\n";
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            usdfile << "        (" << std::scientific << vertices[i][0] << ", " << vertices[i][1] << ", " << vertices[i][2] << ")";
            if (i != vertices.size() - 1)
                usdfile << ",";
            usdfile << "\n";
        }
        usdfile << "    ]\n";
        usdfile << "}\n";
        usdfile.close();
    }
    else if (ext == "wrl")
    {
        std::ofstream wrlfile(filename);
        wrlfile << "#VRML V2.0 utf8\n";
        wrlfile << "Shape {\n";
        wrlfile << "  geometry IndexedFaceSet {\n";
        wrlfile << "    coord Coordinate {\n";
        wrlfile << "      point [\n";
        for (size_t i = 0; i < vertices.size(); ++i)
            wrlfile << "        " << std::scientific << vertices[i] << ",\n";
        wrlfile << "      ]\n";
        wrlfile << "    }\n";
        wrlfile << "    coordIndex [\n";
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            for (size_t j = 0; j < face_indices[i].size(); ++j)
                wrlfile << " " << face_indices[i][j];
            wrlfile << " -1,\n";
        }
        wrlfile << "    ]\n";
        wrlfile << "  }\n";
        wrlfile << "}\n";
        wrlfile.close();
    }
    else if (ext == "x3d")
    {
        std::ofstream x3dfile(filename);
        x3dfile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        x3dfile << "<X3D profile=\"Interchange\" version=\"3.3\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema-instance\" xsd:noNamespaceSchemaLocation=\"http://www.web3d.org/specifications/x3d-3.3.xsd\">\n";
        x3dfile << " <Scene>\n";
        x3dfile << "  <Shape>\n";
        x3dfile << "   <IndexedFaceSet coordIndex=\"";
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            for (size_t j = 0; j < face_indices[i].size(); ++j)
                x3dfile << face_indices[i][j] << " ";
            x3dfile << "-1 ";
        }
        x3dfile << "\">\n";
        x3dfile << "    <Coordinate point=\"";
        for (size_t i = 0; i < vertices.size(); ++i)
            x3dfile << std::scientific << vertices[i] << " ";
        x3dfile << "\"/>\n";
        x3dfile << "   </IndexedFaceSet>\n";
        x3dfile << "  </Shape>\n";
        x3dfile << " </Scene>\n";
        x3dfile << "</X3D>\n";
        x3dfile.close();
    }
    else
        std::cerr << "Unsupported file format: " << ext << '\n';
}
template <typename Real>
void save_mesh(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename)
{
    if (!m_pmesh)
        return;
    // get the file extension
    size_t lastindex = filename.find_last_of(".");
    std::string ext = filename.substr(lastindex + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                   { return std::tolower(c); }); //
    if (ext == "obj")
        m_pmesh->write_obj(filename.c_str());
    else if (ext == "off")
        m_pmesh->write_off(filename.c_str());
    else if (ext == "ply")
        SavePLYmesh_with_float_storage(m_pmesh, filename);
    else
        std::cerr << "Unsupported file format: " << ext << '\n';
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void save_as_svg(MeshLib::Mesh3D<Real> *m_pmesh, const std::vector<std::array<unsigned char, 3>> &ply_face_color, const char svgfilename[])
{
    using namespace LibBoard;
    Board board;
    board.clear();
    Group facegroup;
    auto scale = 500 / std::max(m_pmesh->xmax - m_pmesh->xmin, m_pmesh->ymax - m_pmesh->ymin);
    TinyVector<Real, 3> center((m_pmesh->xmax + m_pmesh->xmin) / 2, (m_pmesh->ymax + m_pmesh->ymin) / 2, (m_pmesh->zmax + m_pmesh->zmin) / 2);

    // if (!ply_face_color.empty())
    {
        board.setLineWidth(0.0);
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); ++i)
        {
            auto face = m_pmesh->get_face(i);
            auto edge = face->edge;
            std::vector<Point> points;
            points.reserve(face->valence);
            do
            {
                const auto p = scale * (edge->pair->vert->pos - center);
                points.emplace_back(p[0], p[1]);
                edge = edge->next;
            } while (edge != face->edge);

            if (!ply_face_color.empty())
            {
                board.setFillColor(Color(ply_face_color[i][0], ply_face_color[i][1], ply_face_color[i][2]));
            }
            else
            {
                board.setFillColor(Color(128, 128, 128));
            }

            board.drawClosedPolyline(points);
        }
    }

    board.saveSVG(svgfilename);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePtsPLY(const std::string &filename, const std::vector<TinyVector<Real, 3>> &points, const std::vector<TinyVector<Real, 3>> *point_normals, const std::vector<Real> *gray_color, bool use_colormap)
{
    std::vector<float> vertexX(points.size()), vertexY(points.size()), vertexZ(points.size()), vertexNormalX, vertexNormalY, vertexNormalZ;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(points.size()); i++)
    {
        vertexX[i] = static_cast<float>(points[i][0]), vertexY[i] = static_cast<float>(points[i][1]), vertexZ[i] = static_cast<float>(points[i][2]);
    }

    if (point_normals)
    {
        vertexNormalX.resize(points.size()), vertexNormalY.resize(points.size()), vertexNormalZ.resize(points.size());
#pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i)
        {
            vertexNormalX[i] = static_cast<float>((*point_normals)[i][0]), vertexNormalY[i] = static_cast<float>((*point_normals)[i][1]), vertexNormalZ[i] = static_cast<float>((*point_normals)[i][2]);
        }
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    if (point_normals)
    {
        plyOut.getElement("vertex").addProperty<float>("nx", vertexNormalX);
        plyOut.getElement("vertex").addProperty<float>("ny", vertexNormalY);
        plyOut.getElement("vertex").addProperty<float>("nz", vertexNormalZ);
    }
    if (gray_color)
    {
        std::vector<std::array<unsigned char, 3>> ply_color(points.size());
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(gray_color->size()); i++)
        {
            auto val = std::min(std::max((*gray_color)[i], Real(0)), Real(1));
            if (!use_colormap)
            {
                ply_color[i][0] = static_cast<unsigned char>(val * 255);
                ply_color[i][1] = static_cast<unsigned char>(val * 255);
                ply_color[i][2] = static_cast<unsigned char>(val * 255);
            }
            else
            {
                const tinycolormap::Color tcolor = tinycolormap::GetColor(val, tinycolormap::ColormapType::Turbo);
                ply_color[i][0] = static_cast<unsigned char>(tcolor.r() * 255.0);
                ply_color[i][1] = static_cast<unsigned char>(tcolor.g() * 255.0);
                ply_color[i][2] = static_cast<unsigned char>(tcolor.b() * 255.0);
            }
        }
        plyOut.addVertexColors(ply_color);
    }
    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage_and_gray_color(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename,
                                                   const std::vector<Real> *gray_color, bool use_face_color, bool use_colormap)
{
    if (!m_pmesh)
        return;

    std::vector<float> vertexX(m_pmesh->get_num_of_vertices()), vertexY(m_pmesh->get_num_of_vertices()), vertexZ(m_pmesh->get_num_of_vertices());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto v = m_pmesh->get_vertex(i);
        vertexX[i] = static_cast<float>(v->pos[0]), vertexY[i] = static_cast<float>(v->pos[1]), vertexZ[i] = static_cast<float>(v->pos[2]);
    }
    std::vector<std::vector<size_t>> faceIndices(m_pmesh->get_num_of_faces());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto f = m_pmesh->get_face(i);
        auto he = f->edge;
        faceIndices[i].reserve(f->valence);
        do
        {
            faceIndices[i].emplace_back(static_cast<size_t>(he->vert->id));
            he = he->next;
        } while (he != f->edge);
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addFaceIndices(faceIndices);

    if (gray_color)
    {

        std::vector<std::array<unsigned char, 3>> ply_color;

        if (use_face_color)
            ply_color.resize(m_pmesh->get_num_of_faces());
        else
            ply_color.resize(m_pmesh->get_num_of_vertices());

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>((*gray_color).size()); i++)
        {
            auto val = std::min(std::max((*gray_color)[i], Real(0)), Real(1));

            if (!use_colormap)
            {
                ply_color[i][0] = static_cast<unsigned char>(val * 255);
                ply_color[i][1] = static_cast<unsigned char>(val * 255);
                ply_color[i][2] = static_cast<unsigned char>(val * 255);
            }
            else
            {
                const tinycolormap::Color tcolor = tinycolormap::GetColor(val, tinycolormap::ColormapType::Turbo);
                ply_color[i][0] = static_cast<unsigned char>(tcolor.r() * 255.0);
                ply_color[i][1] = static_cast<unsigned char>(tcolor.g() * 255.0);
                ply_color[i][2] = static_cast<unsigned char>(tcolor.b() * 255.0);
            }
        }
        if (use_face_color)
            plyOut.addFaceColors(ply_color);
        else
            plyOut.addVertexColors(ply_color);
    }

    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage_and_gray_color(MeshLib::Mesh3D<Real> *m_pmesh, const std::string &filename,
                                                   const std::vector<Real> &gray_color, const Real min_color, const Real max_color, const bool use_face_color)
{
    if (!m_pmesh)
        return;

    std::vector<float> vertexX, vertexY, vertexZ;
    vertexX.resize(m_pmesh->get_num_of_vertices());
    vertexY.resize(m_pmesh->get_num_of_vertices());
    vertexZ.resize(m_pmesh->get_num_of_vertices());
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto v = m_pmesh->get_vertex(i);
        vertexX[i] = static_cast<float>(v->pos[0]), vertexY[i] = static_cast<float>(v->pos[1]), vertexZ[i] = static_cast<float>(v->pos[2]);
    }
    std::vector<std::vector<size_t>> faceIndices;
    faceIndices.reserve(m_pmesh->get_num_of_faces());
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto f = m_pmesh->get_face(i);
        auto he = f->edge;
        std::vector<size_t> vertex_list;
        vertex_list.reserve(f->valence);
        do
        {
            vertex_list.emplace_back(static_cast<size_t>(he->vert->id));
            he = he->next;
        } while (he != f->edge);
        faceIndices.emplace_back(vertex_list);
    }

    std::vector<std::array<unsigned char, 3>> ply_color;

    if (use_face_color)
        ply_color.reserve(m_pmesh->get_num_of_faces());
    else
        ply_color.reserve(m_pmesh->get_num_of_vertices());

    std::array<unsigned char, 3> c;
    for (size_t i = 0; i < gray_color.size(); i++)
    {
        auto val = std::min(std::max((gray_color[i] - min_color) / (max_color - min_color), Real(0)), Real(1));
        auto color = static_cast<unsigned char>(val * 255);
        c[0] = c[1] = c[2] = color;
        ply_color.emplace_back(c);
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addFaceIndices(faceIndices);

    if (!ply_color.empty())
    {
        if (use_face_color)
            plyOut.addFaceColors(ply_color);
        else
            plyOut.addVertexColors(ply_color);
    }

    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYmesh_with_float_storage_and_gray_color(const std::string &filename, const std::vector<TinyVector<Real, 3>> &tri_vertices, const std::vector<ptrdiff_t> &tri_faces, const std::vector<Real> *gray_color, bool use_face_color, bool use_colormap)
{
    std::vector<float> vertexX(tri_vertices.size()), vertexY(tri_vertices.size()), vertexZ(tri_vertices.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(tri_vertices.size()); i++)
    {
        vertexX[i] = static_cast<float>(tri_vertices[i][0]), vertexY[i] = static_cast<float>(tri_vertices[i][1]), vertexZ[i] = static_cast<float>(tri_vertices[i][2]);
    }
    std::vector<std::vector<size_t>> faceIndices(tri_faces.size() / 3);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(tri_faces.size()) / 3; i++)
    {
        faceIndices[i].resize(3);
        faceIndices[i][0] = static_cast<size_t>(tri_faces[i * 3]);
        faceIndices[i][1] = static_cast<size_t>(tri_faces[i * 3 + 1]);
        faceIndices[i][2] = static_cast<size_t>(tri_faces[i * 3 + 2]);
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    plyOut.addFaceIndices(faceIndices);

    if (gray_color)
    {
        std::vector<std::array<unsigned char, 3>> ply_color;

        if (use_face_color)
            ply_color.resize(tri_faces.size() / 3);
        else
            ply_color.resize(tri_vertices.size());

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>((*gray_color).size()); i++)
        {
            auto val = std::min(std::max((*gray_color)[i], Real(0)), Real(1));

            if (!use_colormap)
            {
                ply_color[i][0] = static_cast<unsigned char>(val * 255);
                ply_color[i][1] = static_cast<unsigned char>(val * 255);
                ply_color[i][2] = static_cast<unsigned char>(val * 255);
            }
            else
            {
                const tinycolormap::Color tcolor = tinycolormap::GetColor(val, tinycolormap::ColormapType::Turbo);
                ply_color[i][0] = static_cast<unsigned char>(tcolor.r() * 255.0);
                ply_color[i][1] = static_cast<unsigned char>(tcolor.g() * 255.0);
                ply_color[i][2] = static_cast<unsigned char>(tcolor.b() * 255.0);
            }
        }
        if (use_face_color)
            plyOut.addFaceColors(ply_color);
        else
            plyOut.addVertexColors(ply_color);
    }

    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void SavePLYMesh_with_color(const std::string &filename, const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, const std::vector<size_t> *vertex_cluster_ids, const std::vector<size_t> *face_cluster_ids)
{
    std::vector<float> vertexX(vertices.size()), vertexY(vertices.size()), vertexZ(vertices.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(vertices.size()); i++)
    {
        vertexX[i] = static_cast<float>(vertices[i][0]);
        vertexY[i] = static_cast<float>(vertices[i][1]);
        vertexZ[i] = static_cast<float>(vertices[i][2]);
    }

    std::vector<std::array<unsigned char, 3>> ply_face_color, ply_vertex_color;
    if (face_cluster_ids)
    {
        auto max_cluster_id = *std::max_element(face_cluster_ids->begin(), face_cluster_ids->end());
        auto cluster_colors = generateDistinctColors(static_cast<int>(max_cluster_id) + 1);
        ply_face_color.reserve(face_indices.size());
        for (size_t i = 0; i < face_indices.size(); i++)
        {
            ply_face_color.emplace_back(cluster_colors[(*face_cluster_ids)[i]]);
        }
    }
    if (vertex_cluster_ids)
    {
        auto max_cluster_id = *std::max_element(vertex_cluster_ids->begin(), vertex_cluster_ids->end());
        auto cluster_colors = generateDistinctColors(static_cast<int>(max_cluster_id) + 1);
        ply_vertex_color.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); i++)
        {
            ply_vertex_color.emplace_back(cluster_colors[(*vertex_cluster_ids)[i]]);
        }
    }
    if (!face_cluster_ids && !vertex_cluster_ids)
    {
        // set color according to face degrees
        ply_face_color.reserve(face_indices.size());
        for (size_t i = 0; i < face_indices.size(); i++)
        {
            if (face_indices[i].size() <= 3)
                ply_face_color.emplace_back(std::array<unsigned char, 3>{255, 0, 0});
            else if (face_indices[i].size() == 4)
                ply_face_color.emplace_back(std::array<unsigned char, 3>{230, 230, 230});
            else if (face_indices[i].size() == 5)
                ply_face_color.emplace_back(std::array<unsigned char, 3>{0, 0, 255});
            else if (face_indices[i].size() == 6)
                ply_face_color.emplace_back(std::array<unsigned char, 3>{0, 255, 0});
            else
                ply_face_color.emplace_back(std::array<unsigned char, 3>{255, 255, 0});
        }
    }

    happly::PLYData plyOut;

    plyOut.addElement("vertex", vertexX.size());
    plyOut.getElement("vertex").addProperty<float>("x", vertexX);
    plyOut.getElement("vertex").addProperty<float>("y", vertexY);
    plyOut.getElement("vertex").addProperty<float>("z", vertexZ);
    if (!face_indices.empty())
        plyOut.addFaceIndices(const_cast<std::vector<std::vector<size_t>> &>(face_indices));
    if (!ply_vertex_color.empty())
        plyOut.addVertexColors(ply_vertex_color);
    if (!ply_face_color.empty())
        plyOut.addFaceColors(ply_face_color);

    plyOut.write(filename, happly::DataFormat::Binary);
}
//////////////////////////////////////////////////////////////////////////
template void SavePLYmesh_with_float_storage(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename, const std::vector<ptrdiff_t> *face_cluster_ids, bool use_coloring_algorithm);
template void SaveChartEdge_as_ply(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename, const std::vector<ptrdiff_t> &face_cluster_ids, float normaloffset);
template void SaveMarkedEdge_as_ply(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename, const std::vector<bool> &marked_edge_tag, float normaloffset);
template void SaveEdge_as_ply(const std::string &filename, const std::vector<TinyVector<double, 3>> &vertices, const std::vector<std::pair<size_t, size_t>> &edges);
template void SaveEdge_as_ply(const std::string &filename, const std::vector<MeshLib::HE_edge<double> *> &edges);
template void SaveComponents(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename);
template void save_component_as_ply(const std::vector<MeshLib::HE_face<double> *> &component, const std::string &filename);
template void save_mesh(std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, const std::string &filename);
template void save_mesh(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename);
template void save_as_svg(MeshLib::Mesh3D<double> *m_pmesh, const std::vector<std::array<unsigned char, 3>> &ply_face_color, const char svgfilename[]);
template void SavePtsPLY(const std::string &filename, const std::vector<TinyVector<double, 3>> &points, const std::vector<TinyVector<double, 3>> *point_normals, const std::vector<double> *gray_color, bool use_colormap);
template void SavePLYmesh_with_float_storage_and_gray_color(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename, const std::vector<double> *gray_color, bool use_face_color, bool use_colormap);
template void SavePLYmesh_with_float_storage_and_gray_color(MeshLib::Mesh3D<double> *m_pmesh, const std::string &filename, const std::vector<double> &gray_color, const double min_color, const double max_color, const bool use_face_color);
template void SavePLYmesh_with_float_storage_and_gray_color(const std::string &filename, const std::vector<TinyVector<double, 3>> &tri_vertices, const std::vector<ptrdiff_t> &tri_faces, const std::vector<double> *gray_color, bool use_face_color, bool use_colormap);
template void SavePLYMesh_with_color(const std::string &filename, const std::vector<TinyVector<double, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, const std::vector<size_t> *vertex_cluster_ids, const std::vector<size_t> *face_cluster_ids);
//////////////////////////////////////////////////////////////////////////