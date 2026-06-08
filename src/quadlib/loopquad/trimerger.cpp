#include "trimerger.h"
#include "looputil.h"

//////////////////////////////////////////////////////////////////////////
template <typename Real>
Tri2QuadMerger<Real>::Tri2QuadMerger(MeshLib::Mesh3D<Real> *mesh) : m_pmesh(mesh)
{
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *Tri2QuadMerger<Real>::create_merged_mesh()
{
    if (!m_pmesh)
        return 0;
    MeshLib::Mesh3D<Real> *m_merged_mesh = new MeshLib::Mesh3D<Real>;

    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto v = m_pmesh->get_vertex(i);
        m_merged_mesh->insert_vertex(v->pos);
    }
    size_t num_quads = 0, num_tris = 0;
    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_edges(); i++)
    {
        auto edge = m_pmesh->get_edge(i);
        if (edge->tag == false || edge->id > edge->pair->id || edge->face == 0 || edge->pair->face == 0 || edge->face->tag == false || edge->pair->face->tag == false)
            continue;

        auto v0 = edge->next->vert;
        auto v1 = edge->pair->vert;
        auto v2 = edge->pair->next->vert;
        auto v3 = edge->vert;

        std::vector<MeshLib::HE_vert<Real> *> vertex_list;
        vertex_list.push_back(m_merged_mesh->get_vertex(v0->id));
        vertex_list.push_back(m_merged_mesh->get_vertex(v1->id));
        vertex_list.push_back(m_merged_mesh->get_vertex(v2->id));
        vertex_list.push_back(m_merged_mesh->get_vertex(v3->id));
        m_merged_mesh->insert_face(vertex_list);
        num_quads++;
    }

    for (ptrdiff_t i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto face = m_pmesh->get_face(i);
        if (face->tag == false)
        {
            std::vector<MeshLib::HE_vert<Real> *> vertex_list;
            auto he = face->edge;
            do
            {
                vertex_list.push_back(m_merged_mesh->get_vertex(he->vert->id));
                he = he->next;
            } while (he != face->edge);
            m_merged_mesh->insert_face(vertex_list);
            if (vertex_list.size() == 3)
                num_tris++;
        }
    }

    m_merged_mesh->update_mesh();

    // std::cout << "Number of quads: " << num_quads << std::endl;
    // std::cout << "Number of tris: " << num_tris << std::endl;
    // std::cout << "quad ratio: " << (Real)num_quads / (Real)(num_quads + num_tris) << std::endl;

    return m_merged_mesh;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool Tri2QuadMerger<Real>::edge_mergeable(MeshLib::HE_edge<Real> *edge, bool check_face_tag)
{
    if (edge->tag || edge->pair->tag || edge->face == 0 || edge->pair->face == 0 || edge->face->valence != 3 || edge->pair->face->valence != 3)
        return false;

    if (check_face_tag && (edge->face->tag || edge->pair->face->tag))
        return false;

	auto v0 = edge->next->vert;
	auto v1 = edge->pair->next->vert;
	auto nedge = v0->edge;
    do
    {
		if (nedge->pair->vert == v1)
			return false;
		nedge = nedge->pair->next;
    } while (nedge != v0->edge);

    return true;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
bool Tri2QuadMerger<Real>::quad_mergeable(MeshLib::HE_edge<Real> *edge, Real &angle_regularity,
                                          TinyVector<Real, 3> &normal, TinyVector<Real, 3> &dir0, TinyVector<Real, 3> &dir1, TinyVector<Real, 3> &dir2, TinyVector<Real, 3> &dir3,
                                          bool check_face_tag)
{
    if (!edge_mergeable(edge, check_face_tag))
    {
        return false;
    }

    bool return_tag = true;
    // if the dihedral angle is too small, we do not merge
    Real angle = compute_dihedral_angle(edge);

    if (angle < diheral_angle_bound)
    {
        return_tag = false;
    }

    // if the merged quad has an opposite normal direction, we do not merge
    auto v0 = edge->next->vert;
    auto v1 = edge->pair->vert;
    auto v2 = edge->pair->next->vert;
    auto v3 = edge->vert;

    normal = v0->pos.Cross(v1->pos) + v1->pos.Cross(v2->pos) + v2->pos.Cross(v3->pos) + v3->pos.Cross(v0->pos);

    if (normal.Dot(edge->face->normal) * normal.Dot(edge->pair->face->normal) < 0)
    {
        return_tag = false;
    }

    normal.Normalize();

    // project the merged quad onto the normal plane. If the quad is not convex, we do not merge
    TinyVector<Real, 3> p[4], dir[4];
    TinyVector<Real, 3> cen = (v0->pos + v1->pos + v2->pos + v3->pos) / 4;
    p[0] = v0->pos - (v0->pos - cen).Dot(normal) * normal;
    p[1] = v1->pos - (v1->pos - cen).Dot(normal) * normal;
    p[2] = v2->pos - (v2->pos - cen).Dot(normal) * normal;
    p[3] = v3->pos - (v3->pos - cen).Dot(normal) * normal;
    dir[0] = p[0] - p[1];
    dir[1] = p[1] - p[2];
    dir[2] = p[2] - p[3];
    dir[3] = p[3] - p[0];

    int sign4[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++)
    {
        sign4[i] = normal.Dot(dir[i].Cross(dir[(i + 1) % 4])) > 0 ? 1 : -1;
    }
    if (std::abs(sign4[0] + sign4[1] + sign4[2] + sign4[3]) != 4)
        return_tag = false;

    dir[0].Normalize(), dir[1].Normalize(), dir[2].Normalize(), dir[3].Normalize();

    // compute the angle regularity: \sum_i |angle_i - 90|, and skip low-quality quads
    Real total_angle = 0;
    angle_regularity = 0;
    for (int i = 0; i < 4; i++)
    {
        Real dot_value = dir[i].Dot(dir[(i + 1) % 4]);
        if (compute_angle(std::fabs(dot_value)) < quad_min_angle_bound)
            return_tag = false;
        Real angle = compute_angle(-dot_value);
        if (std::fabs(angle) < angle_bound_for_merging)
            return_tag = false;
        angle_regularity += std::fabs(angle - 90);
        total_angle += angle;
    }
    // if the quad is not convex, we do not merge
    if (total_angle > 390) // concave 390 degrees, set it large than 360 to allow slightly concave quads
        return_tag = false;

    // store the quad edge directions
    dir0 = dir[0], dir1 = dir[1], dir2 = dir[2], dir3 = dir[3];

    return return_tag;
}
//////////////////////////////////////////////////////////////////////////
template class Tri2QuadMerger<double>;