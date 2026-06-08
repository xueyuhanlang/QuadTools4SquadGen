#pragma once
#include "TinyVector.h"
#include <vector>

template <typename Real>
class QuadImprove
{
public:
    QuadImprove(const std::vector<TinyVector<Real, 3>> &input_vertices, const std::vector<std::vector<size_t>> &input_facets);
    void get_improved_mesh(std::vector<TinyVector<Real, 3>> &output_vertices, std::vector<std::vector<size_t>> &output_facets);
    bool improve_mesh();

private:
    std::vector<TinyVector<Real, 3>> vertices;
    std::vector<std::vector<size_t>> facets;
};
