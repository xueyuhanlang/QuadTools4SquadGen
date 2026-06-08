#include "mesh_repair.h"
#include "looputil.h"
#include <chrono>
#include <iostream>
#include "MeshLoader.h"
#include <unordered_set>
#include <limits>

//////////////////////////////////
template <typename Real>
Real handle_degree_6_vertex(MeshLib::HE_vert<Real> *vert,
                            MeshLib::HE_edge<Real> *start_edge)
{
    auto v1 = start_edge->vert, v2 = start_edge->next->vert, v3 = start_edge->next->next->vert;
    start_edge = start_edge->next->next->next->pair;
    auto v4 = start_edge->next->vert, v5 = start_edge->next->next->vert;
    start_edge = start_edge->next->next->next->pair;
    auto v6 = start_edge->next->vert, v7 = start_edge->next->next->vert;
    start_edge = start_edge->next->next->next->pair;
    auto v8 = start_edge->next->vert, v9 = start_edge->next->next->vert;
    start_edge = start_edge->next->next->next->pair;
    auto v10 = start_edge->next->vert, v11 = start_edge->next->next->vert;
    start_edge = start_edge->next->next->next->pair;
    auto v12 = start_edge->next->vert;

    auto a = (v3->pos + v11->pos + v1->pos + v9->pos) / 4;
    auto b = (v3->pos + v7->pos + v5->pos + v9->pos) / 4;

    return quad_regularity(a, v1->pos, v2->pos, v3->pos) + quad_regularity(v3->pos, v4->pos, v5->pos, b) +
           quad_regularity(b, v5->pos, v6->pos, v7->pos) + quad_regularity(b, v7->pos, v8->pos, v9->pos) +
           quad_regularity(a, v9->pos, v10->pos, v11->pos) + quad_regularity(a, v11->pos, v12->pos, v1->pos) +
           quad_regularity(v3->pos, b, v9->pos, a);
}
////////////////////////////////////////////
template <typename Real>
void improve_mesh_topology(std::vector<TinyVector<Real, 3>> &vertices,
                           std::vector<std::vector<size_t>> &polygons)
{

    using namespace std::chrono;

    bool enable644 = true;
    bool remove_degree2 = false;
    bool split_2npolygon = true;
    bool split_hexagon2three = true;
    bool split_hexagon2two = true;
    bool merge_triangles = true;
    bool decompose_pentagon_tri = true;
    bool rotate_adjacent_quad = false;
    bool handle_two_adjacent_quads = false;
    bool repair_degree6 = true;

    auto start = high_resolution_clock::now();

    auto m_quadmesh = create_mesh<Real>(vertices, polygons, true);

    // recreate mesh
    MeshLib::Mesh3D<Real> *new_quad_mesh = new MeshLib::Mesh3D<Real>;
    std::vector<MeshLib::HE_vert<Real> *> vertice_id_map(m_quadmesh->get_num_of_vertices(), 0);
    for (auto i = 0; i < m_quadmesh->get_num_of_vertices(); i++)
    {
        auto vert = m_quadmesh->get_vertex(i);
        vertice_id_map[i] = new_quad_mesh->insert_vertex(vert->pos);
    }

    std::unordered_set<MyEdge<Real>> mesh_edge_set;
    for (auto i = 0; i < m_quadmesh->get_num_of_edges(); i++)
    {
        auto edge = m_quadmesh->get_edge(i);
        if (edge > edge->pair)
            continue;
        mesh_edge_set.insert(MyEdge<Real>(edge->pair->vert, edge->vert));
    }

    m_quadmesh->reset_faces_tag(false);
    m_quadmesh->reset_vertices_tag(false);

    std::vector<int> update_vert_degree(m_quadmesh->get_num_of_vertices(), 0);

    std::vector<int> is_boundary(m_quadmesh->get_num_of_vertices(), 0);
#pragma omp parallel for
    for (auto i = 0; i < m_quadmesh->get_num_of_vertices(); i++)
    {
        auto vert = m_quadmesh->get_vertex(i);
        is_boundary[i] = m_quadmesh->is_on_boundary(vert);
        update_vert_degree[i] = vert->degree;
    }

    std::vector<MeshLib::HE_vert<Real> *> facelist;
    int caseeven = 0, case644 = 0, casedegree2 = 0, casehex = 0, casetri = 0, casepentagon = 0, casedegree6 = 0, casequadflip = 0, casequadinsert = 0;

    if (enable644)
    {
        // handle 644 cases
        for (auto i = 0; i < m_quadmesh->get_num_of_vertices(); i++)
        {
            auto vert = m_quadmesh->get_vertex(i);
            if (vert->tag || update_vert_degree[vert->id] < 4 || is_boundary[vert->id])
                continue;
            auto edge = vert->edge;
            int find = 0;
            do
            {
                if (update_vert_degree[edge->vert->id] == 2 && update_vert_degree[edge->next->next->next->next->vert->id] == 2 &&
                    edge->face->valence == 6 && edge->face->tag == 0 &&
                    edge->pair->face->tag == 0 && edge->pair->face->valence == 4 && edge->pair->face == edge->next->pair->face &&
                    edge->prev->pair->face->tag == 0 && edge->prev->pair->face->valence == 4 &&
                    edge->prev->pair->face == edge->prev->prev->pair->face &&
                    edge->pair->face != edge->prev->prev->pair->face)
                {
                    find = 1;
                    break;
                }
                edge = edge->pair->next;
            } while (edge != vert->edge);

            if (find == 0)
                continue;

            auto v0 = vert, v1 = edge->vert, v2 = edge->next->vert, v3 = edge->next->next->vert;
            auto v4 = edge->next->next->next->vert, v5 = edge->next->next->next->next->vert;
            auto v6 = edge->prev->prev->pair->next->vert;
            auto v7 = edge->pair->next->vert;

            if (v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag || v6->tag || v7->tag)
                continue;

            // case 1: 3412/0721/4601
            auto quality_case1 = quad_regularity(v3->pos, v4->pos, v1->pos, v2->pos) + quad_regularity(v4->pos, v6->pos, v0->pos, v1->pos);
            auto valid_case1 = mesh_edge_set.find(MyEdge<Real>(v1, v4)) == mesh_edge_set.end();
            if (!valid_case1)
                quality_case1 = std::numeric_limits<Real>::max();
            // case 2: 3452/4605/5072
            auto quality_case2 = quad_regularity(v3->pos, v4->pos, v5->pos, v2->pos) + quad_regularity(v5->pos, v0->pos, v7->pos, v2->pos);
            auto valid_case2 = mesh_edge_set.find(MyEdge<Real>(v2, v5)) == mesh_edge_set.end();
            if (!valid_case2)
                quality_case2 = std::numeric_limits<Real>::max();
            // case3: 0723/6034
            auto quality_case3 = quad_regularity(v0->pos, v7->pos, v2->pos, v3->pos) + quad_regularity(v6->pos, v0->pos, v3->pos, v4->pos);
            auto valid_case3 = mesh_edge_set.find(MyEdge<Real>(v0, v3)) == mesh_edge_set.end();
            if (!valid_case3)
                quality_case3 = std::numeric_limits<Real>::max();

            if (quality_case1 <= quality_case2 && quality_case1 <= quality_case3)
            {
                if (!valid_case1)
                    continue;
                facelist = {vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v1->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v4->id], vertice_id_map[v6->id], vertice_id_map[v0->id], vertice_id_map[v1->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v1->degree]++;
                update_vert_degree[v0->degree]--;
                update_vert_degree[v5->degree] = 0;
                v5->tag = true;
                edge->face->tag = edge->prev->pair->face->tag = true;
                mesh_edge_set.insert(MyEdge<Real>(v1, v4));
            }
            else if (quality_case2 <= quality_case1 && quality_case2 <= quality_case3)
            {
                if (!valid_case2)
                    continue;
                facelist = {vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v5->id], vertice_id_map[v0->id], vertice_id_map[v7->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v5->degree]++;
                update_vert_degree[v0->degree]--;
                update_vert_degree[v1->degree] = 0;
                v1->tag = true;
                edge->face->tag = edge->pair->face->tag = true;
                mesh_edge_set.insert(MyEdge<Real>(v2, v5));
            }
            else
            {
                if (!valid_case2)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v7->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v6->id], vertice_id_map[v0->id], vertice_id_map[v3->id], vertice_id_map[v4->id]};
                new_quad_mesh->insert_face(facelist);
                edge->tag = edge->pair->face->tag = edge->prev->pair->face->tag = true;
                v1->tag = v5->tag = true;
                update_vert_degree[v0->id]--;
                update_vert_degree[v2->id]--;
                update_vert_degree[v1->id] = update_vert_degree[v5->id] = 0;
                update_vert_degree[v3->id]++;
                update_vert_degree[v4->id]--;
                mesh_edge_set.insert(MyEdge<Real>(v0, v3));
            }
            case644++;
        }
    }

    if (remove_degree2)
    {
        // remove interior degree-2 vertices
        for (auto i = 0; i < m_quadmesh->get_num_of_vertices(); i++)
        {
            auto vert = m_quadmesh->get_vertex(i);
            if (vert->tag || update_vert_degree[vert->id] != 2 || is_boundary[vert->id])
                continue;
            auto edge = vert->edge;
            if (edge->face->tag || edge->pair->face->tag)
                continue;
            auto v0 = edge->vert, v2 = edge->pair->next->vert;
            if (v0->tag || v2->tag || mesh_edge_set.find(MyEdge<Real>(v0, v2)) != mesh_edge_set.end())
                continue;
            if (edge->face->valence == 4 && edge->pair->face->valence == 4 && v0->degree == 5 && v2->degree == 5 && edge->prev->pair->face == edge->pair->face)
            {
                auto v1 = edge->next->vert, v3 = edge->pair->next->next->vert;

                if (v1->tag || v3->tag)
                    continue;

                edge->face->tag = edge->pair->face->tag = true;
                vert->tag = true;
                update_vert_degree[vert->id] = 0;
                update_vert_degree[v0->id]--;
                update_vert_degree[v2->id]--;

                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
                new_quad_mesh->insert_face(facelist);
                casedegree2++;
                mesh_edge_set.erase(MyEdge<Real>(v0, vert));
                mesh_edge_set.erase(MyEdge<Real>(v2, vert));
                mesh_edge_set.insert(MyEdge<Real>(v0, v2));
            }
            else if (edge->face->valence == 5 && edge->pair->face->valence == 5 && edge->prev->pair->face == edge->pair->face)
            {
                auto v1 = edge->next->vert, v3 = edge->next->next->vert;
                auto v4 = edge->pair->next->next->vert, v5 = edge->pair->next->next->next->vert;
                if (v1->tag || v3->tag || v4->tag || v5->tag)
                    continue;

                vert->tag = true;
                update_vert_degree[vert->id] = 0;
                edge->face->tag = edge->pair->face->tag = true;

                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v3->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v2->id], vertice_id_map[v4->id], vertice_id_map[v5->id], vertice_id_map[v0->id]};
                new_quad_mesh->insert_face(facelist);
                casedegree2++;

                mesh_edge_set.erase(MyEdge<Real>(v0, vert));
                mesh_edge_set.erase(MyEdge<Real>(v2, vert));
                mesh_edge_set.insert(MyEdge<Real>(v0, v2));
            }
        }
    }

    if (split_2npolygon)
    {
        // split 2n-gons polygons into quads, n > 3
        for (auto i = 0; i < m_quadmesh->get_num_of_faces(); i++)
        {
            auto face = m_quadmesh->get_face(i);
            if (face->tag || face->valence % 2 != 0 || face->valence <= 6)
                continue;

            std::vector<MeshLib::HE_vert<Real> *> vertexlist;
            auto edge = face->edge;
            int status = 1;
            do
            {
                vertexlist.push_back(edge->vert);
                if (edge->vert->tag)
                {
                    status = 0;
                    break;
                }
                edge = edge->next;
            } while (edge != face->edge);

            if (status == 0)
                continue;

            TinyVector<Real, 3> case0_center, case1_center;
            for (auto j = 0; j < vertexlist.size(); j += 2)
            {
                case0_center += vertexlist[j]->pos;
                case1_center += vertexlist[j + 1]->pos;
            }
            case0_center /= (Real)(vertexlist.size() / 2);
            case1_center /= (Real)(vertexlist.size() / 2);

            Real quality_case0 = 0, quality_case1 = 0;
            for (auto j = 0; j < vertexlist.size(); j += 2)
            {
                auto v0 = vertexlist[j], v1 = vertexlist[j + 1], v2 = vertexlist[(j + 2) % vertexlist.size()], v3 = vertexlist[(j + 3) % vertexlist.size()];
                quality_case0 += quad_regularity(case1_center, v0->pos, v1->pos, v2->pos);
                quality_case1 += quad_regularity(case0_center, v1->pos, v2->pos, v3->pos);
            }

            if (quality_case0 < quality_case1)
            {
                for (auto j = 0; j < vertexlist.size(); j += 2)
                {
                    auto c = new_quad_mesh->insert_vertex(case1_center);
                    auto v0 = vertexlist[j], v1 = vertexlist[j + 1], v2 = vertexlist[(j + 2) % vertexlist.size()];
                    facelist = {c, vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id]};
                    new_quad_mesh->insert_face(facelist);
                    update_vert_degree[v0->id]++;
                }
            }
            else
            {
                for (auto j = 0; j < vertexlist.size(); j += 2)
                {
                    auto c = new_quad_mesh->insert_vertex(case0_center);
                    auto v1 = vertexlist[j + 1], v2 = vertexlist[(j + 2) % vertexlist.size()], v3 = vertexlist[(j + 3) % vertexlist.size()];
                    facelist = {c, vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
                    new_quad_mesh->insert_face(facelist);
                    update_vert_degree[v1->id]++;
                }
            }
            face->tag = true;
            caseeven++;
        }
    }

    if (split_hexagon2three)
    {
        // split hex into three quads
        for (auto i = 0; i < m_quadmesh->get_num_of_faces(); i++)
        {
            auto face = m_quadmesh->get_face(i);
            if (face->tag || face->valence != 6)
                continue;
            auto edge = face->edge;

            auto v0 = edge->vert, v1 = edge->next->vert, v2 = edge->next->next->vert, v3 = edge->next->next->next->vert, v4 = edge->next->next->next->next->vert, v5 = edge->next->next->next->next->next->vert;

            if (v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag)
                continue;

            if (update_vert_degree[v1->id] == 3 && update_vert_degree[v3->id] == 3 && update_vert_degree[v5->id] == 3)
            {
                auto c = new_quad_mesh->insert_vertex((v1->pos + v3->pos + v5->pos) / (Real)3);
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], c, vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id], c};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v4->id], vertice_id_map[v5->id], c, vertice_id_map[v3->id]};
                new_quad_mesh->insert_face(facelist);
                face->tag = true;
                update_vert_degree[v1->id]++;
                update_vert_degree[v3->id]++;
                update_vert_degree[v5->id]++;
                casehex++;
            }
            else if (update_vert_degree[v0->id] == 3 && update_vert_degree[v2->id] == 3 && update_vert_degree[v4->id] == 3)
            {
                auto c = new_quad_mesh->insert_vertex((v0->pos + v2->pos + v4->pos) / (Real)3);
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], c};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v2->id], vertice_id_map[v3->id], vertice_id_map[v4->id], c};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v4->id], vertice_id_map[v5->id], vertice_id_map[v0->id], c};
                new_quad_mesh->insert_face(facelist);
                face->tag = true;
                update_vert_degree[v0->id]++;
                update_vert_degree[v2->id]++;
                update_vert_degree[v4->id]++;
                casehex++;
            }
        }
    }

    if (split_hexagon2two)
    {
        // split hexagons into two quads, should be excuted after the above hex2threequad step
        for (auto i = 0; i < m_quadmesh->get_num_of_faces(); i++)
        {
            auto face = m_quadmesh->get_face(i);
            if (face->tag || face->valence != 6)
                continue;
            auto edge = face->edge;

            auto v0 = edge->vert, v1 = edge->next->vert, v2 = edge->next->next->vert, v3 = edge->next->next->next->vert, v4 = edge->next->next->next->next->vert, v5 = edge->next->next->next->next->next->vert;

            // if (update_vert_degree[v0->id] != 3 || update_vert_degree[v1->id] != 3 || update_vert_degree[v2->id] != 3 || update_vert_degree[v3->id] != 3 || update_vert_degree[v4->id] != 3 || update_vert_degree[v5->id] != 3 || v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag)
            if (v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag)
                continue;

            // 0123,0345
            Real quality_case0 = quad_regularity(v0->pos, v1->pos, v2->pos, v3->pos) + quad_regularity(v0->pos, v3->pos, v4->pos, v5->pos);
            int valid_case0 = mesh_edge_set.find(MyEdge<Real>(v0, v3)) == mesh_edge_set.end();
            if (!valid_case0)
                quality_case0 = std::numeric_limits<Real>::max();
            // 0145,1234
            Real quality_case1 = quad_regularity(v0->pos, v1->pos, v4->pos, v5->pos) + quad_regularity(v1->pos, v2->pos, v3->pos, v4->pos);
            int valid_case1 = mesh_edge_set.find(MyEdge<Real>(v1, v4)) == mesh_edge_set.end();
            if (!valid_case1)
                quality_case1 = std::numeric_limits<Real>::max();
            // 0125,3452
            Real quality_case2 = quad_regularity(v0->pos, v1->pos, v5->pos, v2->pos) + quad_regularity(v1->pos, v2->pos, v3->pos, v4->pos);
            int valid_case2 = mesh_edge_set.find(MyEdge<Real>(v1, v2)) == mesh_edge_set.end();
            if (!valid_case2)
                quality_case2 = std::numeric_limits<Real>::max();

            if (update_vert_degree[v0->id] > 3 || update_vert_degree[v3->id] > 3)
                quality_case0 = std::numeric_limits<Real>::max();
            if (update_vert_degree[v1->id] > 3 || update_vert_degree[v4->id] > 3)
                quality_case1 = std::numeric_limits<Real>::max();
            if (update_vert_degree[v2->id] > 3 || update_vert_degree[v5->id] > 3)
                quality_case2 = std::numeric_limits<Real>::max();

            if (quality_case0 <= quality_case1 && quality_case0 <= quality_case2)
            {
                if (!valid_case0)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v0->id], vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v0->id]++;
                update_vert_degree[v3->id]++;
                face->tag = true;
                casehex++;
                mesh_edge_set.insert(MyEdge<Real>(v0, v3));
            }
            else if (quality_case1 <= quality_case0 && quality_case1 <= quality_case2)
            {
                if (!valid_case1)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v4->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id], vertice_id_map[v4->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v1->id]++;
                update_vert_degree[v4->id]++;
                face->tag = true;
                casehex++;
                mesh_edge_set.insert(MyEdge<Real>(v1, v4));
            }
            else if (quality_case2 <= quality_case0 && quality_case2 <= quality_case1)
            {
                if (!valid_case2)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v2->id]++;
                update_vert_degree[v5->id]++;
                face->tag = true;
                casehex++;
                mesh_edge_set.insert(MyEdge<Real>(v2, v5));
            }
        }
    }

    // merge adjacent triangles, decompose pentagon_tri faces, rotate adjacent quads
    for (auto i = 0; i < m_quadmesh->get_num_of_edges(); i++)
    {
        auto edge = m_quadmesh->get_edge(i);

        if (edge->face == 0 || edge->pair->face == 0 || edge->face->tag || edge->pair->face->tag)
            continue;
        if (merge_triangles && edge->face->valence == 3 && edge->pair->face->valence == 3)
        {
            // merge two adjacent triangles
            auto v0 = edge->vert, v1 = edge->next->vert, v2 = edge->pair->vert, v3 = edge->pair->next->vert;
            if (update_vert_degree[v0->id] > 4 && update_vert_degree[v2->id] > 4)
            {
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
                new_quad_mesh->insert_face(facelist);
                edge->face->tag = edge->pair->face->tag = true;
                update_vert_degree[v0->id]--;
                update_vert_degree[v2->id]--;
                casetri++;
                mesh_edge_set.erase(MyEdge<Real>(v0, v2));
            }
        }
        else if (decompose_pentagon_tri && edge->face->valence == 3 && edge->pair->face->valence == 5)
        {

            auto v0 = edge->next->vert, v1 = edge->pair->vert, v5 = edge->vert;
            auto v2 = edge->pair->next->vert, v3 = edge->pair->next->next->vert, v4 = edge->pair->next->next->next->vert;

            if (v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag)
                continue;

            // 0123,0345
            Real quality_case0 = quad_regularity(v0->pos, v1->pos, v2->pos, v3->pos) + quad_regularity(v0->pos, v3->pos, v4->pos, v5->pos);
            int valid_case0 = mesh_edge_set.find(MyEdge<Real>(v0, v3)) == mesh_edge_set.end();
            if (update_vert_degree[v0->id] == 2 || update_vert_degree[v3->id] == 2)
                quality_case0 = -1;
            if (!valid_case0)
                quality_case0 = 2;
            // 0145,1234
            Real quality_case1 = quad_regularity(v0->pos, v1->pos, v4->pos, v5->pos) + quad_regularity(v1->pos, v2->pos, v3->pos, v4->pos);
            int valid_case1 = mesh_edge_set.find(MyEdge<Real>(v1, v4)) == mesh_edge_set.end();
            if (update_vert_degree[v1->id] == 2 || update_vert_degree[v4->id] == 2)
                quality_case1 = -1;
            if (!valid_case1)
                quality_case1 = 2;
            // 0125,3452
            Real quality_case2 = quad_regularity(v0->pos, v1->pos, v5->pos, v2->pos) + quad_regularity(v1->pos, v2->pos, v3->pos, v4->pos);
            int valid_case2 = mesh_edge_set.find(MyEdge<Real>(v2, v5)) == mesh_edge_set.end();
            if (update_vert_degree[v2->id] == 2 || update_vert_degree[v5->id] == 2)
                quality_case2 = -1;
            if (!valid_case2)
                quality_case2 = 2;

            if (quality_case0 <= quality_case1 && quality_case0 <= quality_case2)
            {
                if (!valid_case0)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v0->id], vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v0->id]++;
                update_vert_degree[v3->id]++;
                mesh_edge_set.erase(MyEdge<Real>(v1, v5));
                mesh_edge_set.insert(MyEdge<Real>(v0, v3));
            }
            else if (quality_case1 <= quality_case0 && quality_case1 <= quality_case2)
            {
                if (!valid_case1)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v4->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id], vertice_id_map[v4->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v1->id]++;
                update_vert_degree[v4->id]++;
                mesh_edge_set.erase(MyEdge<Real>(v1, v5));
                mesh_edge_set.insert(MyEdge<Real>(v1, v4));
            }
            else
            {
                if (!valid_case2)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v2->id]++;
                update_vert_degree[v5->id]++;
                mesh_edge_set.erase(MyEdge<Real>(v1, v5));
                mesh_edge_set.insert(MyEdge<Real>(v2, v5));
            }
            update_vert_degree[v1->id]--;
            update_vert_degree[v5->id]--;
            edge->face->tag = edge->pair->face->tag = true;
            casepentagon++;
        }
        else if (rotate_adjacent_quad && edge->face->valence == 4 && edge->pair->face->valence == 4)
        {
            // quad rotation
            auto v0 = edge->pair->vert, v3 = edge->vert, v4 = edge->next->vert, v5 = edge->next->next->vert;
            auto v1 = edge->pair->next->vert, v2 = edge->pair->next->next->vert;

            if (update_vert_degree[v0->id] <= 4 || update_vert_degree[v3->id] <= 4 || v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag)
                continue;

            // 0123,0345
            Real quality_case0 = quad_regularity(v0->pos, v1->pos, v2->pos, v3->pos) + quad_regularity(v0->pos, v3->pos, v4->pos, v5->pos);
            // 0145,1234
            Real quality_case1 = quad_regularity(v0->pos, v1->pos, v4->pos, v5->pos) + quad_regularity(v1->pos, v2->pos, v3->pos, v4->pos);
            int valid_case1 = mesh_edge_set.find(MyEdge<Real>(v1, v4)) == mesh_edge_set.end();
            if (!valid_case1)
                quality_case1 = 2;
            // 0125,3452
            Real quality_case2 = quad_regularity(v0->pos, v1->pos, v2->pos, v5->pos) + quad_regularity(v2->pos, v3->pos, v4->pos, v5->pos);
            int valid_case2 = mesh_edge_set.find(MyEdge<Real>(v2, v5)) == mesh_edge_set.end();
            if (!valid_case2)
                quality_case2 = 2;

            if (quality_case0 <= quality_case1 && quality_case0 <= quality_case2)
            {
                // do nothing
                continue;
            }
            else if (quality_case1 <= quality_case0 && quality_case1 <= quality_case2)
            {
                if (!valid_case1)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v4->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id], vertice_id_map[v4->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v1->id]++;
                update_vert_degree[v4->id]++;
                update_vert_degree[v0->id]--;
                update_vert_degree[v3->id]--;
                mesh_edge_set.erase(MyEdge<Real>(v0, v3));
                mesh_edge_set.insert(MyEdge<Real>(v1, v4));
            }
            else
            {
                if (!valid_case2)
                    continue;
                facelist = {vertice_id_map[v0->id], vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v5->id]};
                new_quad_mesh->insert_face(facelist);
                facelist = {vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id], vertice_id_map[v2->id]};
                new_quad_mesh->insert_face(facelist);
                update_vert_degree[v2->id]++;
                update_vert_degree[v5->id]++;
                update_vert_degree[v0->id]--;
                update_vert_degree[v3->id]--;
                mesh_edge_set.erase(MyEdge<Real>(v0, v3));
                mesh_edge_set.insert(MyEdge<Real>(v2, v5));
            }
            edge->face->tag = edge->pair->face->tag = true;
            casequadflip++;
        }
    }

    if (handle_two_adjacent_quads)
    {
        // handle two adjacent quads via vertex insertion
        for (auto i = 0; i < m_quadmesh->get_num_of_edges(); i++)
        {
            continue; // this function is not correct
            auto edge = m_quadmesh->get_edge(i);
            if (edge->face == 0 || edge->pair->face == 0 || edge->face->tag || edge->pair->face->tag || edge->face->valence != 4 || edge->pair->face->valence != 4)
                continue;

            auto v0 = edge->pair->vert, v1 = edge->vert, v2 = edge->next->vert, v3 = edge->next->next->vert;
            auto v4 = edge->pair->next->vert, v5 = edge->pair->next->next->vert;

            if (v0->tag || v1->tag || v2->tag || v3->tag || v4->tag || v5->tag || update_vert_degree[v2->id] != 3 || update_vert_degree[v5->id] != 3 || update_vert_degree[v1->id] < 4 || update_vert_degree[v0->id] < 4)
                continue;

            auto c = new_quad_mesh->insert_vertex((v0->pos + v1->pos) / 2);
            facelist = {vertice_id_map[v5->id], vertice_id_map[v1->id], vertice_id_map[v2->id], c};
            new_quad_mesh->insert_face(facelist);
            facelist = {vertice_id_map[v2->id], vertice_id_map[v3->id], vertice_id_map[v0->id], c};
            new_quad_mesh->insert_face(facelist);
            facelist = {vertice_id_map[v0->id], vertice_id_map[v4->id], vertice_id_map[v5->id], c};
            new_quad_mesh->insert_face(facelist);

            casequadinsert++;
            edge->face->tag = edge->pair->face->tag = true;
        }
    }

    if (repair_degree6)
    {
        // repair degree-6 vertex's surrounding region
        for (auto i = 0; i < m_quadmesh->get_num_of_vertices(); i++)
        {
            // continue;
            auto vert = m_quadmesh->get_vertex(i);
            if (vert->tag || vert->degree != 6 || is_boundary[vert->id])
                continue;
            auto edge = vert->edge;
            int status = 1;
            MeshLib::HE_edge<Real> *start_edge = 0;
            Real quality = std::numeric_limits<Real>::max();
            Real orginal_quality = 0;
            do
            {
                if (edge->face->tag || edge->face->valence != 4)
                {
                    status = 0;
                    break;
                }

                orginal_quality += quad_regularity(edge->vert->pos, edge->next->vert->pos, edge->next->next->vert->pos, edge->next->next->next->vert->pos);

                if (update_vert_degree[edge->vert->id] == 4 && update_vert_degree[edge->pair->next->pair->next->pair->next->vert->id] == 4)
                {
                    if (start_edge == 0)
                    {
                        start_edge = edge;
                        quality = handle_degree_6_vertex(vert, edge);
                    }
                    else
                    {
                        auto new_quality = handle_degree_6_vertex(vert, edge);
                        if (new_quality < quality)
                        {
                            quality = new_quality;
                            start_edge = edge;
                        }
                    }
                }
                edge = edge->pair->next;
            } while (edge != vert->edge);
            if (status == 0 || start_edge == 0)
                continue;

            orginal_quality /= 6;
            quality /= 7;
            if (quality > orginal_quality)
                continue;

            auto v1 = start_edge->vert, v2 = start_edge->next->vert, v3 = start_edge->next->next->vert;
            start_edge = start_edge->next->next->next->pair;
            auto v4 = start_edge->next->vert, v5 = start_edge->next->next->vert;
            start_edge = start_edge->next->next->next->pair;
            auto v6 = start_edge->next->vert, v7 = start_edge->next->next->vert;
            start_edge = start_edge->next->next->next->pair;
            auto v8 = start_edge->next->vert, v9 = start_edge->next->next->vert;
            start_edge = start_edge->next->next->next->pair;
            auto v10 = start_edge->next->vert, v11 = start_edge->next->next->vert;
            start_edge = start_edge->next->next->next->pair;
            auto v12 = start_edge->next->vert;

            if (v1->tag || v2->tag || v3->tag || v4->tag || v5->tag || v6->tag || v7->tag || v8->tag || v9->tag || v10->tag || v11->tag || v12->tag)
                continue;

            auto a = new_quad_mesh->insert_vertex((v3->pos + v11->pos + v1->pos + v9->pos) / 4);
            auto b = new_quad_mesh->insert_vertex((v3->pos + v7->pos + v5->pos + v9->pos) / 4);

            facelist = {a, vertice_id_map[v1->id], vertice_id_map[v2->id], vertice_id_map[v3->id]};
            new_quad_mesh->insert_face(facelist);

            facelist = {vertice_id_map[v3->id], vertice_id_map[v4->id], vertice_id_map[v5->id], b};
            new_quad_mesh->insert_face(facelist);

            facelist = {b, vertice_id_map[v5->id], vertice_id_map[v6->id], vertice_id_map[v7->id]};
            new_quad_mesh->insert_face(facelist);

            facelist = {vertice_id_map[v7->id], vertice_id_map[v8->id], vertice_id_map[v9->id], b};
            new_quad_mesh->insert_face(facelist);

            facelist = {vertice_id_map[v9->id], vertice_id_map[v10->id], vertice_id_map[v11->id], a};
            new_quad_mesh->insert_face(facelist);

            facelist = {a, vertice_id_map[v11->id], vertice_id_map[v12->id], vertice_id_map[v1->id]};
            new_quad_mesh->insert_face(facelist);

            facelist = {vertice_id_map[v3->id], b, vertice_id_map[v9->id], a};
            new_quad_mesh->insert_face(facelist);

            update_vert_degree[v3->id]++;
            update_vert_degree[v9->id]++;
            vert->tag = true;
            do
            {
                edge->face->tag = true;
                edge = edge->pair->next;
            } while (edge != vert->edge);

            mesh_edge_set.erase(MyEdge<Real>(vert, v1));
            mesh_edge_set.erase(MyEdge<Real>(vert, v3));
            mesh_edge_set.erase(MyEdge<Real>(vert, v5));
            mesh_edge_set.erase(MyEdge<Real>(vert, v7));
            mesh_edge_set.erase(MyEdge<Real>(vert, v9));
            mesh_edge_set.erase(MyEdge<Real>(vert, v11));

            casedegree6++;
        }
    }

    for (auto i = 0; i < m_quadmesh->get_num_of_faces(); i++)
    {
        auto face = m_quadmesh->get_face(i);
        if (face->tag)
            continue;
        facelist.resize(0);
        auto edge = face->edge;
        do
        {
            facelist.push_back(vertice_id_map[edge->pair->vert->id]);
            edge = edge->next;
        } while (edge != face->edge);
        if (facelist.size() > 2)
            new_quad_mesh->insert_face(facelist);
    }

    new_quad_mesh->update_mesh();
    delete m_quadmesh;
    m_quadmesh = new_quad_mesh;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    int total = caseeven + case644 + casedegree2 + casehex + casetri + casepentagon + casedegree6 + casequadflip + casequadinsert;
    if (total > 0)
    {
        std::cout << "mesh repair (" << duration.count() << " ms): ";
        if (caseeven > 0)
            std::cout << caseeven << " even cases, ";
        if (case644 > 0)
            std::cout << case644 << " 644 cases, ";
        if (casedegree2 > 0)
            std::cout << casedegree2 << " degree-2 cases, ";
        if (casehex > 0)
            std::cout << casehex << " hexagon cases, ";
        if (casetri > 0)
            std::cout << casetri << " triangle cases, ";
        if (casepentagon > 0)
            std::cout << casepentagon << " pentagon cases, ";
        if (casedegree6 > 0)
            std::cout << casedegree6 << " degree-6 cases, ";
        if (casequadflip > 0)
            std::cout << casequadflip << " quad flip cases, ";
        if (casequadinsert > 0)
            std::cout << casequadinsert << " quad insert cases, ";

        std::cout << std::endl;
    }
    else
        std::cout << "mesh repair: no cases handled." << std::endl;

    // m_quadmesh->write_obj("improved_mesh.obj");
    mesh_to_vertices_and_faces(m_quadmesh, vertices, polygons);
    delete m_quadmesh;
}

//////////////////////////////////
template void improve_mesh_topology(std::vector<TinyVector<double, 3>> &vertices,
                                    std::vector<std::vector<size_t>> &polygons);
template double handle_degree_6_vertex(MeshLib::HE_vert<double> *vert,
                                       MeshLib::HE_edge<double> *start_edge);