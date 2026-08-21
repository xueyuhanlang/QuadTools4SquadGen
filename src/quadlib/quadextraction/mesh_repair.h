#pragma once

#include "Mesh3D.h"

template <typename Real>
class MyEdge
{
public:
    MyEdge(MeshLib::HE_vert<Real> *u0 = nullptr, MeshLib::HE_vert<Real> *u1 = nullptr) noexcept
    {
        if (u0 < u1)
        {
            v[0] = u0, v[1] = u1;
        }
        else
        {
            v[0] = u1, v[1] = u0;
        }
    }
    //////////////////////////////////////////////////////////////////////////
    [[nodiscard]] bool operator==(const MyEdge &p) const noexcept
    {
        return v[0] == p.v[0] && v[1] == p.v[1];
    }
    //////////////////////////////////////////////////////////////////////////
    [[nodiscard]] bool operator!=(const MyEdge &p) const noexcept
    {
        return !(*this == p);
    }
    //////////////////////////////////////////////////////////////////////////
    [[nodiscard]] bool operator<(const MyEdge &p) const noexcept
    {
        for (int i = 0; i < 2; i++)
        {
            if (v[i] < p.v[i])
                return true;
            else if (v[i] > p.v[i])
                return false;
        }
        return false;
    }

public:
    MeshLib::HE_vert<Real> *v[2];
};

namespace std
{
    template <typename Real>
    struct hash<MyEdge<Real>>
    {
        std::size_t operator()(const MyEdge<Real> &e) const noexcept
        {
            std::size_t seed = 0;
            seed ^= std::hash<ptrdiff_t>{}(e.v[0]->id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<ptrdiff_t>{}(e.v[1]->id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

template <typename Real>
void improve_mesh_topology(std::vector<TinyVector<Real, 3>> &vertices,
                           std::vector<std::vector<size_t>> &polygons);

template <typename Real>
Real handle_degree_6_vertex(MeshLib::HE_vert<Real> *vert, MeshLib::HE_edge<Real> *start_edge);