#include "NonmanifoldProcess.h"
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <functional>
#include "PointUtil.h"

////////////////////////////////////////////////
bool has_nonmanifold_issue(size_t num_vertices, const std::vector<std::vector<size_t>> &face_indices)
{
    std::unordered_map<std::pair<size_t, size_t>, std::pair<ptrdiff_t, ptrdiff_t>> edge_map;
    for (ptrdiff_t f = 0; f < (ptrdiff_t)face_indices.size(); f++)
    {
        const auto &face = face_indices[f];

        std::unordered_set<size_t> s(face.begin(), face.end());

        if (face.size() != s.size())
            return true;

        for (size_t i = 0; i < face.size(); i++)
        {
            auto v0 = face[i];
            auto v1 = face[(i + 1) % face.size()];
            if (v0 > v1)
                std::swap(v0, v1);
            auto edge = std::make_pair(v0, v1);
            auto eiter = edge_map.find(edge);
            if (eiter == edge_map.end())
                edge_map[edge] = std::make_pair(f, -1);
            else if (eiter->second.second == -1)
                eiter->second.second = f;
            else
                return true; // non-manifold edge
        }
    }

    std::vector<std::vector<std::pair<size_t, size_t>>> vertex_ring(num_vertices);
    std::vector<int> vertex_boundary_degree(num_vertices, 0);
    for (const auto &edge : edge_map)
    {
        vertex_ring[edge.first.first].emplace_back(edge.first);
        vertex_ring[edge.first.second].emplace_back(edge.first);
        if (edge.second.second == -1)
        {
            vertex_boundary_degree[edge.first.first]++;
            vertex_boundary_degree[edge.first.second]++;
            if (vertex_boundary_degree[edge.first.first] > 2 || vertex_boundary_degree[edge.first.second] > 2)
                return true; // non-manifold vertex
        }
    }

    for (size_t i = 0; i < vertex_ring.size(); i++)
    {
        if (vertex_ring[i].size() < 2)
            continue;
        auto e = vertex_ring[i].front();
        std::unordered_set<std::pair<size_t, size_t>> visited_edges;
        std::queue<std::pair<size_t, size_t>> edge_queue;
        edge_queue.push(e);
        while (!edge_queue.empty())
        {
            auto edge = edge_queue.front();
            edge_queue.pop();
            if (visited_edges.find(edge) != visited_edges.end())
                continue;
            visited_edges.insert(edge);
            const auto &face_pair = edge_map[edge];
            ptrdiff_t f[2] = {face_pair.first, face_pair.second};
            for (int j = 0; j < 2; j++)
            {
                if (f[j] == -1)
                    continue;
                const auto &face = face_indices[f[j]];
                for (int k = 0; k < (int)face.size(); k++)
                {
                    auto v0 = face[k];
                    auto v1 = face[(k + 1) % face.size()];
                    if (v0 > v1)
                        std::swap(v0, v1);
                    if (v0 == i || v1 == i)
                    {
                        auto next_edge = std::make_pair(v0, v1);
                        edge_queue.push(next_edge);
                    }
                }
            }
        }
        if (visited_edges.size() < vertex_ring[i].size())
            return true; // non-manifold vertex
    }

    return false;
}
////////////////////////////////////////////////
template <typename Real>
bool merge_boundary_vertices(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &fInd, bool &nonmanifold_issue, bool fast_mode, bool skip_degenerate_faces)
{
    std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>> edge2facemap;
    for (auto i = 0; i < fInd.size(); i++)
    {
        auto &face = fInd[i];
        for (auto j = 0; j < face.size(); j++)
        {
            auto v0 = face[j];
            auto v1 = face[(j + 1) % face.size()];
            auto edge = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
            edge2facemap[edge].emplace_back(i);
        }
    }
    std::vector<bool> boundary_vertex_tags(vertices.size(), false);
    size_t num_boundary_edges = 0;
    for (const auto &edge_faces : edge2facemap)
    {
        if (edge_faces.second.size() != 2)
        {
            boundary_vertex_tags[edge_faces.first.first] = true;
            boundary_vertex_tags[edge_faces.first.second] = true;
            num_boundary_edges++;
        }
    }

    if (num_boundary_edges == 0)
    {
        nonmanifold_issue = has_nonmanifold_issue(vertices.size(), fInd);
        return false;
    }

    std::vector<TinyVector<Real, 3>> mesh_boundary_points;
    std::vector<size_t> mesh_boundary_vertices_ids;
    mesh_boundary_points.reserve(2 * num_boundary_edges);
    mesh_boundary_vertices_ids.reserve(2 * num_boundary_edges);
    for (auto v_id = 0; v_id < vertices.size(); v_id++)
    {
        if (boundary_vertex_tags[v_id])
        {
            mesh_boundary_vertices_ids.emplace_back(v_id);
            mesh_boundary_points.emplace_back(vertices[v_id]);
        }
    }

    std::vector<ptrdiff_t> merge2uniqueID_map, back2overlapID_map;
    size_t unique_pt_counter = MergeSamePoints<Real>(mesh_boundary_points, merge2uniqueID_map, back2overlapID_map);
    bool merged = false;
    if (unique_pt_counter < mesh_boundary_points.size())
    {
        std::vector<TinyVector<Real, 3>> merged_vertices;
        std::vector<std::vector<size_t>> merged_faces;
        merged_vertices.reserve(vertices.size()), merged_vertices.resize(0);
        merged_faces.reserve(fInd.size()), merged_faces.resize(0);

        std::vector<size_t> vertex_id_map(vertices.size(), 0);
        for (size_t i = 0; i < vertices.size(); i++)
        {
            if (boundary_vertex_tags[i])
                continue;
            merged_vertices.emplace_back(vertices[i]);
            vertex_id_map[i] = merged_vertices.size() - 1;
        }
        size_t num_interior_vertices = merged_vertices.size();
        for (size_t i = 0; i < (size_t)unique_pt_counter; i++)
        {
            auto id = back2overlapID_map[i];
            merged_vertices.emplace_back(mesh_boundary_points[id]);
        }
        for (size_t i = 0; i < merge2uniqueID_map.size(); i++)
        {
            vertex_id_map[mesh_boundary_vertices_ids[i]] = merge2uniqueID_map[i] + num_interior_vertices;
        }

        for (const auto &face : fInd)
        {
            std::vector<size_t> new_face;
            new_face.reserve(face.size());
            for (const auto v_id : face)
            {
                new_face.emplace_back(vertex_id_map[v_id]);
            }
            std::unordered_set<size_t> s(new_face.begin(), new_face.end());
            if (!skip_degenerate_faces || s.size() > 2)
                merged_faces.emplace_back(new_face);
        }

        if (fast_mode)
        {
            vertices = merged_vertices;
            fInd = merged_faces;
            merged = true;
        }
        else
        {
            if (!has_nonmanifold_issue(merged_vertices.size(), merged_faces))
            {
                vertices = merged_vertices;
                fInd = merged_faces;
                nonmanifold_issue = false;
                return true; // successfully merged boundary vertices
            }
            else
            {
                merged = false;
            }
        }
    }
    nonmanifold_issue = has_nonmanifold_issue(vertices.size(), fInd);
    return merged;
}
////////////////////////////////////////////////
template <typename Real>
void manifold_submesh_extraction(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &fInd, const bool input_is_manifold, bool fast_mode)
{
    std::vector<size_t> face_submesh_ids(fInd.size(), 0);
    std::vector<size_t> id_map(vertices.size(), 0);
    // build connection map
    std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>> edge2facemap;
    for (auto i = 0; i < fInd.size(); i++)
    {
        auto &face = fInd[i];
        for (auto j = 0; j < face.size(); j++)
        {
            auto v0 = face[j];
            auto v1 = face[(j + 1) % face.size()];
            auto edge = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
            edge2facemap[edge].emplace_back(i);
        }
    }
    std::vector<bool> boundary_vertex_tags(vertices.size(), false);
    bool has_boundary = false;
    for (const auto &edge_faces : edge2facemap)
    {
        if (edge_faces.second.size() != 2)
        {
            boundary_vertex_tags[edge_faces.first.first] = true;
            boundary_vertex_tags[edge_faces.first.second] = true;
            has_boundary = true;
        }
    }

    if (input_is_manifold && !has_boundary)
    {
        // no submesh extraction needed
        return;
    }

    std::vector<bool> face_visited(fInd.size(), false);
    int submesh_id = 0;

    std::vector<TinyVector<Real, 3>> total_submesh_vertices;
    std::vector<std::vector<size_t>> total_submesh_faces;
    total_submesh_vertices.reserve(vertices.size());
    total_submesh_faces.reserve(fInd.size());

    size_t num_updated_facets = 0;
    for (auto fi = 0; fi < fInd.size(); fi++)
    {
        if (face_visited[fi])
            continue;
        std::unordered_set<size_t> submesh_vertice_ids, submesh_facet_ids;
        std::queue<size_t> face_queue;
        face_visited[fi] = true;
        face_queue.push(fi);
        while (!face_queue.empty())
        {
            auto new_face_id = face_queue.front();
            face_submesh_ids[new_face_id] = submesh_id;
            face_queue.pop();
            submesh_facet_ids.insert(new_face_id);
            submesh_vertice_ids.insert(fInd[new_face_id].begin(), fInd[new_face_id].end());
            auto &face = fInd[new_face_id];
            for (int j = 0; j < (int)face.size(); j++)
            {
                auto v0 = face[j];
                auto v1 = face[(j + 1) % face.size()];
                auto edge = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
                const auto &neighbors = edge2facemap[edge];
                for (auto neighbor_face_id : neighbors)
                {
                    if (face_visited[neighbor_face_id])
                        continue;
                    else
                    {
                        face_visited[neighbor_face_id] = true;
                        face_queue.push(neighbor_face_id);
                    }
                }
            }
        }
        submesh_id++;

        // if (submesh_facet_ids.size() % 2 != 0)
        //     continue;

        // process submesh
        std::vector<TinyVector<Real, 3>> submesh_vertices, submesh_boundary_points;
        std::vector<size_t> submesh_boundary_vertices_ids;
        std::vector<std::vector<size_t>> new_submesh_faces;
        std::vector<bool> submesh_boundary_tags;

        submesh_vertices.reserve(submesh_vertice_ids.size());
        submesh_boundary_vertices_ids.reserve(submesh_vertice_ids.size());
        submesh_boundary_points.reserve(submesh_vertice_ids.size());
        submesh_boundary_tags.reserve(submesh_vertice_ids.size());
        new_submesh_faces.reserve(submesh_facet_ids.size());

        id_map.assign(vertices.size(), 0);
        for (const auto v_id : submesh_vertice_ids)
        {
            submesh_vertices.emplace_back(vertices[v_id]);
            id_map[v_id] = submesh_vertices.size() - 1;
            if (boundary_vertex_tags[v_id])
            {
                submesh_boundary_vertices_ids.emplace_back(id_map[v_id]);
                submesh_boundary_points.emplace_back(submesh_vertices.back());
                submesh_boundary_tags.emplace_back(true);
            }
            else
            {
                submesh_boundary_tags.emplace_back(false);
            }
        }
        std::vector<size_t> new_face;
        new_face.reserve(3);
        for (const auto f_id : submesh_facet_ids)
        {
            new_face.resize(0);
            for (const auto v_id : fInd[f_id])
                new_face.emplace_back(id_map[v_id]);
            new_submesh_faces.emplace_back(new_face);
        }
        // check whether the submesh is manifold
        bool nonmanifold = has_nonmanifold_issue(submesh_vertices.size(), new_submesh_faces);

        if (!fast_mode)
        {
            // glue boundary vertices when possible
            if (submesh_boundary_vertices_ids.size() > 1 && new_submesh_faces.size() > 1)
            {
                std::vector<ptrdiff_t> merge2uniqueID_map, back2overlapID_map;
                auto unique_pt_counter = MergeSamePoints<Real>(submesh_boundary_points, merge2uniqueID_map, back2overlapID_map);

                if ((size_t)unique_pt_counter < submesh_boundary_points.size())
                {
                    // create a new submesh
                    std::vector<TinyVector<Real, 3>> new_submesh_vertices;
                    std::vector<std::vector<size_t>> new_submesh_faces2;
                    new_submesh_vertices.reserve(submesh_vertices.size());
                    new_submesh_faces2.reserve(new_submesh_faces.size());
                    std::vector<size_t> vertex_id_map(submesh_vertices.size(), 0);
                    for (size_t i = 0; i < submesh_vertices.size(); i++)
                    {
                        if (submesh_boundary_tags[i])
                            continue;
                        new_submesh_vertices.emplace_back(submesh_vertices[i]);
                        vertex_id_map[i] = new_submesh_vertices.size() - 1;
                    }
                    size_t num_interior_vertices = new_submesh_vertices.size();
                    for (size_t i = 0; i < (size_t)unique_pt_counter; i++)
                    {
                        auto id = back2overlapID_map[i];
                        // auto bnd_index = submesh_boundary_vertices_ids[id];
                        new_submesh_vertices.emplace_back(submesh_boundary_points[id]);
                        // vertex_id_map[bnd_index] = new_submesh_vertices.size() - 1;
                    }
                    for (size_t i = 0; i < merge2uniqueID_map.size(); i++)
                    {
                        vertex_id_map[submesh_boundary_vertices_ids[i]] = merge2uniqueID_map[i] + num_interior_vertices;
                    }
                    for (const auto &face : new_submesh_faces)
                    {
                        std::vector<size_t> new_face;
                        new_face.reserve(face.size());
                        for (const auto v_id : face)
                        {
                            new_face.emplace_back(vertex_id_map[v_id]);
                        }
                        std::unordered_set<size_t> s(new_face.begin(), new_face.end());
                        if (s.size() > 2)
                            new_submesh_faces2.emplace_back(new_face);
                    }

                    if (!has_nonmanifold_issue(new_submesh_vertices.size(), new_submesh_faces2))
                    {
                        submesh_vertices = new_submesh_vertices;
                        new_submesh_faces = new_submesh_faces2;
                        // std::swap(submesh_vertices, new_submesh_vertices);
                        // std::swap(new_submesh_faces, new_submesh_faces2);
                        nonmanifold = false;
                    }
                }
            }
        }
        if (!nonmanifold)
        {
            // add to total submesh vertices and faces
            for (const auto &new_face : new_submesh_faces)
            {
                auto update_face = new_face;
                for (auto &v_id : update_face)
                {
                    v_id += total_submesh_vertices.size();
                }
                total_submesh_faces.emplace_back(update_face);
            }

            total_submesh_vertices.insert(total_submesh_vertices.end(),
                                          submesh_vertices.begin(),
                                          submesh_vertices.end());
        }
    }

    if (fast_mode == false && total_submesh_faces.size() > 0 && total_submesh_vertices.size() < vertices.size())
    {
        bool nonmanifold;
        merge_boundary_vertices<Real>(total_submesh_vertices, total_submesh_faces, nonmanifold, fast_mode);
    }

    // update original vertices and faces
    if (total_submesh_vertices.size() < vertices.size())
    {
        vertices = total_submesh_vertices;
        fInd = total_submesh_faces;
    }
}
////////////////////////////////////////////////
template <typename Real>
bool nonmanifold_merge(std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, bool fast_mode)
{
    if (face_indices.empty())
        return false; // no valid faces

    bool has_nonmanifold = false;
    merge_boundary_vertices<Real>(vertices, face_indices, has_nonmanifold, fast_mode);
    manifold_submesh_extraction<Real>(vertices, face_indices, !has_nonmanifold, fast_mode);
    return !face_indices.empty();
}
////////////////////////////////////////////////
template <typename Real>
size_t label_connected_components(const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, std::vector<size_t> &face_submesh_ids, const size_t start_id)
{
    face_submesh_ids.assign(face_indices.size(), start_id);
    std::vector<size_t> id_map(vertices.size(), 0);
    // build connection map
    std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>> edge2facemap;
    for (auto i = 0; i < face_indices.size(); i++)
    {
        auto &face = face_indices[i];
        for (auto j = 0; j < face.size(); j++)
        {
            auto v0 = face[j];
            auto v1 = face[(j + 1) % face.size()];
            auto edge = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
            edge2facemap[edge].emplace_back(i);
        }
    }
    auto submesh_id = start_id;
    std::vector<bool> face_visited(face_indices.size(), false);
    for (auto fi = 0; fi < face_indices.size(); fi++)
    {
        if (face_visited[fi])
            continue;
        std::queue<size_t> face_queue;
        face_visited[fi] = true;
        face_queue.push(fi);
        while (!face_queue.empty())
        {
            auto new_face_id = face_queue.front();
            face_submesh_ids[new_face_id] = submesh_id;
            face_queue.pop();
            auto &face = face_indices[new_face_id];
            for (int j = 0; j < (int)face.size(); j++)
            {
                auto v0 = face[j];
                auto v1 = face[(j + 1) % face.size()];
                auto edge = v0 < v1 ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
                const auto &neighbors = edge2facemap[edge];
                for (auto neighbor_face_id : neighbors)
                {
                    if (face_visited[neighbor_face_id])
                        continue;
                    else
                    {
                        face_visited[neighbor_face_id] = true;
                        face_queue.push(neighbor_face_id);
                    }
                }
            }
        }
        submesh_id++;
    }
    return submesh_id;
}
////////////////////////////////////////////////
template bool merge_boundary_vertices(std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &fInd, bool &nonmanifold_issue, bool fast_mode, bool skip_degenerate_faces);
template void manifold_submesh_extraction(std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &fInd, const bool input_is_manifold, bool fast_mode);
template bool nonmanifold_merge(std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, bool fast_mode);
template size_t label_connected_components(const std::vector<TinyVector<double, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, std::vector<size_t> &face_submesh_ids, const size_t start_id);