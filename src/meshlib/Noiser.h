#pragma once

#include "TinyVector.h"

template <typename Real, int FeatureDim>
class Noiser
{
public:
    Noiser(int x_grid_size = 8, int y_grid_size = 8, int z_grid_size = 8);
    void get_perlin_noise(const Real x, const Real y, const Real z, TinyVector<Real, FeatureDim> &noise);
    void get_random_noise(TinyVector<Real, FeatureDim> &noise);
    Real get_random_noise();
    void shuffle_perlin_cube();

protected:
    int x_size, y_size, z_size;
    std::vector<TinyVector<Real, FeatureDim>> perlin_cube;
};