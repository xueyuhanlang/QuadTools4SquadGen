#include "loopmerger.h"

#include <limits>
#include <queue>
#include <deque>
#include <unordered_set>
#include "looputil.h"

//////////////////////////////////////////////////////////////////////////
template <typename Real>
LoopMerger<Real>::LoopMerger(MeshLib::Mesh3D<Real> *mesh, int smooth_num) : Tri2QuadMerger<Real>(mesh), laplacian_smooth_iter(smooth_num)
{
    if (mesh)
    {
        m_pmesh = mesh;
        mesh->reset_all_tag(false);
        backup_positions.resize(m_pmesh->get_num_of_vertices());
        for (ptrdiff_t i = 0; i < mesh->get_num_of_vertices(); i++)
            backup_positions[i] = mesh->get_vertex(i)->pos;
        loop_merge();
    }
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *LoopMerger<Real>::get_merged_mesh()
{
    return this->create_merged_mesh();
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void LoopMerger<Real>::merge_edge(MeshLib::HE_edge<Real> *edge)
{
    edge->tag = edge->pair->tag = true;
    edge->face->tag = edge->pair->face->tag = true;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::opposite_to_singularity(MeshLib::HE_edge<Real> *edge)
{
    return (edge->face && edge->next->vert->degree > 8) || (edge->pair->face && edge->pair->next->vert->degree > 8);
}
template <typename Real>
bool LoopMerger<Real>::connected_with_singularity(MeshLib::HE_edge<Real> *edge)
{
    return edge->vert->degree > 8 || edge->pair->vert->degree > 8;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void LoopMerger<Real>::loop_merge()
{
    // assume m_pmesh is manifold
    edge_mergeable_tag.assign(m_pmesh->get_num_of_edges(), false);
    edge_quad_regularities.assign(m_pmesh->get_num_of_edges(), 0);
    edge_quad_normals.assign(m_pmesh->get_num_of_edges(), TinyVector<Real, 3>(0, 0, 0));
    edge_quad_directions.assign(4 * m_pmesh->get_num_of_edges(), TinyVector<Real, 3>(0, 0, 0));

    std::vector<std::pair<ptrdiff_t, Real>> edge_scores;
    edge_scores.reserve(m_pmesh->get_num_of_edges() / 2);
    m_pmesh->reset_edges_tag(false);
    std::pair<ptrdiff_t, Real> edge_pair;
    TinyVector<Real, 3> quad_normal, dir0, dir1, dir2, dir3;

    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (i > edge->pair->id)
            continue;
        Real angle_regularity = 0;
        if (this->quad_mergeable(edge, angle_regularity, quad_normal, dir0, dir1, dir2, dir3, false))
        {
            edge_pair.first = i;
            edge_pair.second = angle_regularity;
            edge_mergeable_tag[i] = edge_mergeable_tag[edge->pair->id] = true;
            edge_quad_regularities[i] = edge_quad_regularities[edge->pair->id] = angle_regularity;
            edge_quad_normals[i] = edge_quad_normals[edge->pair->id] = quad_normal;
            edge_quad_directions[4 * i] = dir0;
            edge_quad_directions[4 * i + 1] = dir1;
            edge_quad_directions[4 * i + 2] = dir2;
            edge_quad_directions[4 * i + 3] = dir3;
            edge_quad_directions[4 * edge->pair->id] = dir2;
            edge_quad_directions[4 * edge->pair->id + 1] = dir3;
            edge_quad_directions[4 * edge->pair->id + 2] = dir0;
            edge_quad_directions[4 * edge->pair->id + 3] = dir1;
            edge_scores.push_back(edge_pair);
        }
        edge->tag = edge->pair->tag = true;
    }
    // sort edge_scores according to the angle regularity
    std::sort(edge_scores.begin(), edge_scores.end(), [](const std::pair<ptrdiff_t, Real> &a, const std::pair<ptrdiff_t, Real> &b)
              { return a.second < b.second; });

    //////////////////////////////////////////////////////////////////////////
    // start growing loops
    m_pmesh->reset_edges_tag(false);
    m_pmesh->reset_faces_tag(false);

    for (auto edge_score : edge_scores)
    {
        auto s_edge = m_pmesh->get_edge(edge_score.first);

        if (!this->edge_mergeable(s_edge))
            continue;

        edge_queue.push(std::make_pair(s_edge, (Real)0));

        while (!edge_queue.empty())
        {
            auto edge = edge_queue.top().first;
            edge_queue.pop();

            if (!this->edge_mergeable(edge) || opposite_to_singularity(edge) || connected_with_singularity(edge))
                continue;

            merge_edge(edge);

            const auto &normal = edge_quad_normals[edge->id];

            const auto &v01 = edge_quad_directions[4 * edge->id];
            const auto &v12 = edge_quad_directions[4 * edge->id + 1];
            const auto &v23 = edge_quad_directions[4 * edge->id + 2];
            const auto &v30 = edge_quad_directions[4 * edge->id + 3];

            auto dir_1 = edge->next;
            auto dir_2 = edge->next->next;
            auto dir_3 = edge->pair->next;
            auto dir_4 = edge->pair->next->next;

            insert_edge_for_merging(normal, dir_1, v23, v01);
            insert_edge_for_merging(normal, dir_2, v30, v12);
            insert_edge_for_merging(normal, dir_3, v01, v23);
            insert_edge_for_merging(normal, dir_4, v12, v30);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // connectivity correction
    loop_shifting();
    tri_merge(true);
    bool cleanup_tag = false;
    do
    {
        cleanup_tag = cleanup();
        // if (cleanup_tag)
        //     std::cout << "cleanup done" << std::endl;
    } while (cleanup_tag);
}
template <typename Real>
bool LoopMerger<Real>::trigger_degree_2_vertex(MeshLib::HE_edge<Real> *edge)
{
    bool boundary_vertex_tag = false;
    return (vertex_valid_degree(edge->vert, boundary_vertex_tag) <= 3) || (vertex_valid_degree(edge->pair->vert, boundary_vertex_tag) <= 3);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::tri_merge(bool disable_edge_connected_with_singularity)
{
    bool trimerged = false;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge->tag == false && edge_mergeable_tag[edge->id] && edge->face->tag == false && edge->pair->face->tag == false)
        {
            if (disable_edge_connected_with_singularity && connected_with_singularity(edge))
                continue;
            edge_queue.push(std::make_pair(edge, quadflow_alignment_score(edge)));
        }
    }

    bool boundary_vertex_tag = false;
    while (!edge_queue.empty())
    {
        auto edge_pair = edge_queue.top();
        auto edge = edge_pair.first;
        edge_queue.pop();
        if (edge->tag || edge->face->tag || edge->pair->face->tag)
            continue;
        if (trigger_degree_2_vertex(edge))
            continue;
        merge_edge(edge);
        trimerged = true;
    }

    return trimerged;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::local_shift(bool disable_edge_opposite_to_singularity)
{
    bool shifted = false;

    bool shift = true;
    do
    {
        shift = false;
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
        {
            auto edge = m_pmesh->get_edge(i);
            if (edge->tag == false && edge_mergeable_tag[edge->id] && edge->face->tag == false && edge->pair->face->tag)
            {
                if (disable_edge_opposite_to_singularity && opposite_to_singularity(edge))
                    continue;
                edge_queue.push(std::make_pair(edge, quadflow_alignment_score(edge)));
            }
        }

        while (!edge_queue.empty())
        {
            auto edge_pair = edge_queue.top();
            auto edge = edge_pair.first;
            edge_queue.pop();
            if (edge->tag || edge->face->tag || edge->pair->face->tag == false)
                continue;

            int e_count = 0;
            Real e_score = quadflow_alignment_score(edge, &e_count);

            auto opposite_edge = edge->pair;

            Real alignment_score = 0;
            int opp_count = 0;
            if (opposite_edge->next->tag)
            {
                alignment_score = quadflow_alignment_score(opposite_edge->next, &opp_count);
                if (alignment_score > e_score)
                    opposite_edge = opposite_edge->next;
            }
            else if (opposite_edge->next->next->tag)
            {
                alignment_score = quadflow_alignment_score(opposite_edge->next->next, &opp_count);
                if (alignment_score > e_score)
                    opposite_edge = opposite_edge->next->next;
            }

            if (opposite_edge != edge->pair)
            {
                opposite_edge->tag = opposite_edge->pair->tag = false;
                opposite_edge->face->tag = opposite_edge->pair->face->tag = false;
                edge->tag = edge->pair->tag = true;
                edge->face->tag = edge->pair->face->tag = true;
                int new_count = 0, new_quad_count = 0;
                Real new_e_score = quadflow_alignment_score(edge, &new_count);
                Real new_alignment_score = quadflow_alignment_score(opposite_edge, &new_quad_count);

                if (new_count + new_quad_count > e_count + opp_count ||
                    ((new_count + new_quad_count == e_count + opp_count) && new_e_score + new_alignment_score < e_score + alignment_score))
                {
                    shift = true;
                    shifted = true;
                    edge_queue.push(std::make_pair(opposite_edge, new_alignment_score));
                }
                else
                {
                    opposite_edge->tag = opposite_edge->pair->tag = true;
                    opposite_edge->face->tag = opposite_edge->pair->face->tag = true;
                    edge->tag = edge->pair->tag = false;
                    edge->face->tag = false;
                }
            }
        }
    } while (shift);

    return shifted;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void LoopMerger<Real>::insert_edge_for_merging(const TinyVector<Real, 3> &normal, MeshLib::HE_edge<Real> *dir, const TinyVector<Real, 3> &v01, const TinyVector<Real, 3> &v23, const Real angle_bound)
{
    if (dir->pair->face && dir->pair->face->tag == false && dir->pair->face->valence == 3)
    {
        auto ne_0 = dir->pair->next;
        auto ne_1 = dir->pair->next->next;
        Real angle_difference_0 = 0, angle_difference_1 = 0;
        Real regularity_0 = 0, regularity_1 = 0;
        bool p0 = merge_possibility(ne_0, true, normal, v01, v23, angle_difference_0, regularity_0);
        bool p1 = merge_possibility(ne_1, false, normal, v01, v23, angle_difference_1, regularity_1);
        if (p0 && p1)
        {
            if (angle_difference_0 < angle_difference_1)
            {
                if (angle_difference_0 < angle_bound)
                    edge_queue.push(std::make_pair(ne_0, regularity_0));
            }
            else
            {
                if (angle_difference_1 < angle_bound)
                    edge_queue.push(std::make_pair(ne_1, regularity_1));
            }
        }
        else if (p0 && !p1)
        {
            if (angle_difference_0 < angle_bound)
                edge_queue.push(std::make_pair(ne_0, regularity_0));
        }
        else if (!p0 && p1)
        {
            if (angle_difference_1 < angle_bound)
                edge_queue.push(std::make_pair(ne_1, regularity_1));
        }
    }
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::merge_possibility(MeshLib::HE_edge<Real> *edge, bool dir_switch, const TinyVector<Real, 3> &normal, const TinyVector<Real, 3> &ref_v01, const TinyVector<Real, 3> &ref_v23, Real &angle_difference, Real &regularity)
{
    if (edge_mergeable_tag[edge->id] == false || edge->tag || edge->pair->tag || edge->face->tag || edge->pair->face->tag)
        return false;

    auto v0 = edge->next->vert;
    auto v1 = edge->pair->vert;
    auto v2 = edge->pair->next->vert;
    auto v3 = edge->vert;

    regularity = edge_quad_regularities[edge->id];

    auto v01 = dir_switch ? (v2->pos - v1->pos) : (v0->pos - v1->pos);
    v01 = v01 - v01.Dot(normal) * normal;
    v01.Normalize();
    auto v23 = dir_switch ? ((v0->pos - v3->pos)) : (v2->pos - v3->pos);
    v23 = v23 - v23.Dot(normal) * normal;
    v23.Normalize();

    auto angle_0 = compute_angle(std::fabs(v01.Dot(ref_v01)));
    auto angle_1 = compute_angle(std::fabs(v23.Dot(ref_v23)));

    angle_difference = (angle_0 + angle_1) / 2;

    return true;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
Real LoopMerger<Real>::quadflow_alignment_score(MeshLib::HE_edge<Real> *edge, int *num_neighbor_quads)
{
    MeshLib::HE_edge<Real> *ne[4];
    ne[0] = edge->next->next;       // e10
    ne[1] = edge->pair->next;       // e21
    ne[2] = edge->pair->next->next; // e32
    ne[3] = edge->next;             // e03

    Real score = 0;
    int count = 0;
    for (int i = 0; i < 4; i++)
    {
        const auto &dir0 = edge_quad_directions[4 * edge->id + (i + 3) % 4];
        const auto &dir1 = edge_quad_directions[4 * edge->id + (i + 1) % 4];

        auto opposite_edge = ne[i]->pair;
        if (opposite_edge->next->tag)
        {
            const auto &ndir0 = edge_quad_directions[4 * opposite_edge->next->id + 1];
            const auto &ndir1 = edge_quad_directions[4 * opposite_edge->next->id + 3];
            score += compute_angle(std::fabs(dir0.Dot(ndir0))) + compute_angle(std::fabs(dir1.Dot(ndir1)));
            count++;
        }
        else if (opposite_edge->next->next->tag)
        {
            const auto &ndir0 = edge_quad_directions[4 * opposite_edge->next->next->id];
            const auto &ndir1 = edge_quad_directions[4 * opposite_edge->next->next->id + 2];
            score += compute_angle(std::fabs(dir0.Dot(ndir0))) + compute_angle(std::fabs(dir1.Dot(ndir1)));
            count++;
        }
    }
    if (num_neighbor_quads)
        *num_neighbor_quads = count;
    return count > 0 ? score / count : 360;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::loop_shifting()
{
    std::vector<bool> edge_changed_tag(m_pmesh->get_num_of_edges(), false);

    bool loop_shifted = true;
    bool occured = false;
    int count = 0;
    while (loop_shifted)
    {
        loop_shifted = false;
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
        {
            auto edge = m_pmesh->get_edge(i);

            if (valid_for_loop_shifting(edge, count == 0))
            {
                edge_queue.push(std::make_pair(edge, quadflow_alignment_score(edge)));
            }
        }

        while (!edge_queue.empty())
        {
            auto edge = edge_queue.top().first;
            edge_queue.pop();

            if (!valid_for_loop_shifting(edge, count == 0))
                continue;

            if (edge_changed_tag[edge->id])
                continue;

            // loop growing
            std::unordered_set<MeshLib::HE_edge<Real> *> old_loop_edge_set, new_loop_edge_set;
            new_loop_edge_set.insert(edge);

            auto start_edge = edge;
            bool valid_loop = false;
            while (1)
            {
                if (start_edge->pair->face == 0)
                    break;

                if (start_edge->pair->face->tag == false)
                {
                    valid_loop = valid_for_loop_shifting(start_edge->pair, count == 0);
                    break;
                }
                else
                {
                    if (start_edge->pair->next->tag)
                    {
                        auto old_loop_edge = start_edge->pair->next;

                        if (edge_changed_tag[old_loop_edge->id])
                            break;
                        if (old_loop_edge_set.find(old_loop_edge) != old_loop_edge_set.end())
                            break;

                        auto new_loop_edge = old_loop_edge->pair->next->next;

                        if (edge_changed_tag[new_loop_edge->id])
                            break;

                        if (count == 0 && !edge_mergeable_tag[new_loop_edge->id])
                            break;
                        if (new_loop_edge_set.find(new_loop_edge) != new_loop_edge_set.end())
                            break;

                        old_loop_edge_set.insert(old_loop_edge);
                        new_loop_edge_set.insert(new_loop_edge);

                        start_edge = new_loop_edge;
                    }
                    else if (start_edge->pair->next->next->tag)
                    {
                        auto old_loop_edge = start_edge->pair->next->next;
                        if (edge_changed_tag[old_loop_edge->id])
                            break;
                        if (old_loop_edge_set.find(old_loop_edge) != old_loop_edge_set.end())
                            break;

                        auto new_loop_edge = old_loop_edge->pair->next;
                        if (edge_changed_tag[new_loop_edge->id])
                            break;
                        if (count == 0 && !edge_mergeable_tag[new_loop_edge->id])
                            break;
                        if (new_loop_edge_set.find(new_loop_edge) != new_loop_edge_set.end())
                            break;

                        old_loop_edge_set.insert(old_loop_edge);
                        new_loop_edge_set.insert(new_loop_edge);

                        start_edge = new_loop_edge;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (!valid_loop)
                continue;

            for (auto edge : old_loop_edge_set)
            {
                edge->tag = edge->pair->tag = false;
                edge->face->tag = edge->pair->face->tag = false;
                edge_changed_tag[edge->id] = edge_changed_tag[edge->pair->id] = true;
            }
            for (auto edge : new_loop_edge_set)
            {
                edge->tag = edge->pair->tag = true;
                edge->face->tag = edge->pair->face->tag = true;
                edge_changed_tag[edge->id] = edge_changed_tag[edge->pair->id] = true;
            }

            loop_shifted = true;
            occured = true;
        }
    }
    return occured;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
int LoopMerger<Real>::vertex_valid_degree(MeshLib::HE_vert<Real> *vert, bool &boundary_vertex_tag)
{
    auto edge = vert->edge;
    int valid_edge_degree = 0;
    boundary_vertex_tag = false;
    do
    {
        if (edge->tag == false)
            valid_edge_degree++;
        if (edge->face == 0)
            boundary_vertex_tag = true;
        edge = edge->pair->next;
    } while (edge != vert->edge);
    return valid_edge_degree;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::valid_for_loop_shifting(MeshLib::HE_edge<Real> *edge, const bool relaxed)
{
    if (edge->face == 0 || edge->face->valence != 3 || edge->face->tag || edge->tag || edge->pair->tag)
        return false;

    if (!edge_mergeable_tag[edge->id])
        return false;

    auto vert = edge->next->vert;
    bool boundary_vertex_tag = false;
    int valid_edge_degree = vertex_valid_degree(vert, boundary_vertex_tag);
    if (boundary_vertex_tag)
    {
        if (valid_edge_degree > 4)
            return false;
    }
    else
    {
        return (valid_edge_degree == 4) || (valid_edge_degree == 5);
    }
    return true;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void LoopMerger<Real>::vertex_smoothing(int max_iter)
{
    if (max_iter < 1)
        return;

    vertex_smooth_valence.assign(m_pmesh->get_num_of_vertices(), 0);

    for (int iter = 0; iter < max_iter; iter++)
    {
        new_positions.assign(m_pmesh->get_num_of_vertices(), TinyVector<Real, 3>(0, 0, 0));
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
        {
            auto edge = m_pmesh->get_edge(i);
            if (edge->tag == true || edge->id > edge->pair->id)
                continue;
            auto v0 = edge->vert;
            auto v1 = edge->pair->vert;
            new_positions[v0->id] += v1->pos;
            new_positions[v1->id] += v0->pos;
            if (iter == 0)
            {
                vertex_smooth_valence[v0->id]++;
                vertex_smooth_valence[v1->id]++;
            }
        }

        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
        {
            auto v = m_pmesh->get_vertex(i);
            if (vertex_smooth_valence[i] > 0)
                v->pos = new_positions[i] / vertex_smooth_valence[i];
        }
    }
    m_pmesh->update_normal(true);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool LoopMerger<Real>::cleanup()
{
    bool occured = false;
    bool find = true;

    while (find)
    {
        find = false;
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
        {
            auto edge = m_pmesh->get_edge(i);
            if (edge->tag == true && edge->face && edge->pair->face && edge->face->valence == 3 && edge->pair->face->valence == 3)
            {
                auto ne0 = edge->next;
                auto ne1 = edge->pair->next->next;
                auto ne2 = edge->pair->next;
                auto ne3 = edge->next->next;

                if (ne0->pair->face && ne1->pair->face && ne0->pair->face->tag == false && ne1->pair->face->tag == false && trigger_degree_2_vertex(ne0) == false && trigger_degree_2_vertex(ne1) == false)
                {
                    corner_case_merge(edge, ne0, ne1);
                    find = true;
                    occured = true;
                }
                else if (ne0->pair->face && ne2->pair->face && ne0->pair->face->tag == false && ne2->pair->face->tag == false && trigger_degree_2_vertex(ne0) == false && trigger_degree_2_vertex(ne2) == false)
                {
                    corner_case_merge(edge, ne0, ne2);
                    find = true;
                    occured = true;
                }
                else if (ne1->pair->face && ne3->pair->face && ne1->pair->face->tag == false && ne3->pair->face->tag == false && trigger_degree_2_vertex(ne1) == false && trigger_degree_2_vertex(ne3) == false)
                {
                    corner_case_merge(edge, ne1, ne3);
                    find = true;
                    occured = true;
                }
            }
        }
    }

    find = true;
    while (find)
    {
        find = false;
        for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
        {
            auto edge = m_pmesh->get_edge(i);
            if (edge->tag == false && edge->face && edge->pair->face && edge->face->valence == 3 && edge->pair->face->valence == 3 && edge->face->tag == false && edge->pair->face->tag == false && trigger_degree_2_vertex(edge) == false)
            {
                merge_edge(edge);
                find = true;
                occured = true;
            }
        }
    }

    return occured;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void LoopMerger<Real>::corner_case_merge(MeshLib::HE_edge<Real> *edge, MeshLib::HE_edge<Real> *e0, MeshLib::HE_edge<Real> *e1)
{
    e0->tag = e0->pair->tag = e1->tag = e1->pair->tag = true;
    e0->face->tag = e0->pair->face->tag = e1->face->tag = e1->pair->face->tag = true;
    edge->tag = edge->pair->tag = false;
}
//////////////////////////////////////////////////////////////////////////

template class LoopMerger<double>;