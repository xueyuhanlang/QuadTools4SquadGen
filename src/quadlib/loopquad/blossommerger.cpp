#include "blossommerger.h"
#include "PerfectMatching.h"
#include "GEOM/GeomPerfectMatching.h"
#include "MyTuple.h"

template <typename Real>
BlossomMerger<Real>::BlossomMerger(MeshLib::Mesh3D<Real> *mesh) : Tri2QuadMerger<Real>(mesh)
{
    if (this->m_pmesh)
    {
        blossom_merge();
    }
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *BlossomMerger<Real>::get_merged_mesh()
{
    return valid_compute ? this->create_merged_mesh() : nullptr;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void BlossomMerger<Real>::blossom_merge()
{
    valid_compute = false;
    auto m_pmesh = this->m_pmesh;
    m_pmesh->reset_all_tag(false);
    const int node_num = static_cast<int>(m_pmesh->get_num_of_faces());
    int edge_num = 0;
    std::vector<int> edges;
    std::vector<int> weights;
    std::unordered_map<MySortedTuple<ptrdiff_t, 2, false>, MeshLib::HE_edge<Real> *> interior_edges;
    edges.reserve(m_pmesh->get_num_of_edges());
    weights.reserve(m_pmesh->get_num_of_edges() / 2);

    std::pair<ptrdiff_t, Real> edge_pair;
    TinyVector<Real, 3> quad_normal, dir0, dir1, dir2, dir3;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge->face == nullptr || edge->pair->face == nullptr || edge->pair->face->id > edge->face->id)
            continue;
        Real angle_regularity = 0;
        bool suc = this->quad_mergeable(edge, angle_regularity, quad_normal, dir0, dir1, dir2, dir3, false);
		//bool suc = this->edge_mergeable(edge, false);
        if (!suc)
            angle_regularity = 100000; // set to a large value to skip this edge

        edges.emplace_back(static_cast<int>(edge->pair->face->id));
        edges.emplace_back(static_cast<int>(edge->face->id));
        weights.emplace_back(static_cast<int>(100 * angle_regularity));
        interior_edges[MySortedTuple<ptrdiff_t, 2, false>(edge->pair->face->id, edge->face->id)] = edge;
        ++edge_num;
    }

    PerfectMatching *pm = new PerfectMatching(node_num, edge_num);

    for (int e = 0; e < edge_num; ++e)
        pm->AddEdge(edges[2 * e], edges[2 * e + 1], weights[e]);

    try
    {

        pm->Solve();
        int res = CheckPerfectMatchingOptimality(node_num, edge_num, edges.data(), weights.data(), pm);
        // std::cout << ((res == 0) ? "ok" : ((res == 1) ? "error" : "fatal error")) << std::endl;
        // double cost = ComputePerfectMatchingCost(node_num, edge_num, edges, weights, pm);
        // std::cout << "Perfect matching cost: " << cost << std::endl;

        for (int i = 0; i < node_num; ++i)
        {
            auto j = pm->GetMatch(i);
            if (i > j)
                continue;
            auto edge = interior_edges[MySortedTuple<ptrdiff_t, 2, false>(i, j)];
            edge->tag = edge->pair->tag = true;
            edge->face->tag = edge->pair->face->tag = true;
        }
        valid_compute = true;
    }
    catch (...)
    {
        m_pmesh->reset_all_tag(false);
    }

    delete pm;
}
//////////////////////////////////////////////////////////////////////////
template class BlossomMerger<double>;