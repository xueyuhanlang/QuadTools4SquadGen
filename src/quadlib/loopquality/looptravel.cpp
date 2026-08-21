#include "looptravel.h"

////////////////////////////////////////////////
template <typename Real>
bool edge_loop_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_edge<Real> *> &edge_loop, const std::vector<bool> &singular_vertex_tag, std::vector<bool> &complex_edge_tag, bool tag_complex)
{
    if (edge == nullptr || edge->tag || edge->pair->tag)
        return false;

    std::vector<MeshLib::HE_edge<Real> *> edge_loop_left, edge_loop_right;

    edge_travel(edge, edge_loop_left, singular_vertex_tag);
    edge_travel(edge->pair, edge_loop_right, singular_vertex_tag);

    edge_loop.clear();
    edge_loop.reserve(edge_loop_left.size() + edge_loop_right.size() + 1);
    for (auto e = edge_loop_right.rbegin(); e != edge_loop_right.rend(); ++e)
    {
        edge_loop.emplace_back((*e)->pair);
    }
    edge_loop.emplace_back(edge);
    edge_loop.insert(edge_loop.end(), edge_loop_left.begin(), edge_loop_left.end());

    if (tag_complex)
    {
        for (const auto *e : edge_loop)
            complex_edge_tag[e->id] = complex_edge_tag[e->pair->id] = true;
    }
    return true;
}
////////////////////////////////////////////////
template <typename Real>
void edge_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_edge<Real> *> &edge_path, const std::vector<bool> &singular_vertex_tag)
{
    auto e = edge;
    edge->tag = edge->pair->tag = true;
    do
    {
        if (singular_vertex_tag[e->vert->id])
        {
            break;
        }
        if (e->face == nullptr || e->pair->face == nullptr)
        {
            auto e2 = e->pair;
            do
            {
                if (e2 != e->pair && (e2->face == nullptr || e2->pair->face == nullptr))
                {
                    e = e2;
                    break;
                }
                e2 = e2->pair->next;
            } while (e2 != e->pair);
        }
        else
        {
            if (e->vert->degree == 4)
                e = e->next->pair->next;
            else
                break;
        }
        if (e->tag || e->pair->tag)
        {
            return;
        }
        if (e != edge)
        {
            edge_path.emplace_back(e);
            e->tag = e->pair->tag = true;
        }
    } while (e != edge);
}
////////////////////////////////////////////////
template <typename Real>
void face_loop_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_face<Real> *> &face_loop)
{
    std::vector<MeshLib::HE_face<Real> *> face_loop_left, face_loop_right;
    face_travel(edge, face_loop_right);
    face_travel(edge->pair, face_loop_left);
    face_loop.clear();
    face_loop.reserve(face_loop_left.size() + face_loop_right.size());
    for (auto f = face_loop_left.rbegin(); f != face_loop_left.rend(); ++f)
    {
        face_loop.emplace_back(*f);
    }
    face_loop.insert(face_loop.end(), face_loop_right.begin(), face_loop_right.end());
}
////////////////////////////////////////////////
template <typename Real>
void face_travel(MeshLib::HE_edge<Real> *edge, std::vector<MeshLib::HE_face<Real> *> &face_path)
{
    auto e = edge;
    do
    {
        auto e_next2 = e->next->next;
        if (e->tag || e_next2->tag || e->face == nullptr)
            break;
        face_path.emplace_back(e->face);
        e->tag = e_next2->tag = true;
        e = e_next2->pair;
    } while (e != edge);
}
////////////////////////////////////////////////
template bool edge_loop_travel<double>(MeshLib::HE_edge<double> *edge, std::vector<MeshLib::HE_edge<double> *> &edge_loop, const std::vector<bool> &singular_vertex_tag, std::vector<bool> &complex_edge_tag, bool tag_complex);
template void edge_travel<double>(MeshLib::HE_edge<double> *edge, std::vector<MeshLib::HE_edge<double> *> &edge_path, const std::vector<bool> &singular_vertex_tag);
template void face_loop_travel<double>(MeshLib::HE_edge<double> *edge, std::vector<MeshLib::HE_face<double> *> &face_loop);
template void face_travel<double>(MeshLib::HE_edge<double> *edge, std::vector<MeshLib::HE_face<double> *> &face_path);
