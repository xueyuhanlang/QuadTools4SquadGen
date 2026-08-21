#include "Noiser.h"
#include <cstdlib>
#include <iostream>

template <typename Real, int FeatureDim>
Noiser<Real, FeatureDim>::Noiser(int x_grid_size, int y_grid_size, int z_grid_size)
    : x_size(x_grid_size), y_size(y_grid_size), z_size(z_grid_size)
{
    x_size = std::max(3, x_size);
    y_size = std::max(3, y_size);
    z_size = std::max(3, z_size);
    perlin_cube.resize(x_size * y_size * z_size);
    shuffle_perlin_cube();
}

template <typename Real, int FeatureDim>
void Noiser<Real, FeatureDim>::get_perlin_noise(const Real x, const Real y, const Real z, TinyVector<Real, FeatureDim> &noise)
{
    // x, y, z are in [0, 1]
    // get eight corners of the cube
    Real mx = x * (x_size - 1), my = y * (y_size - 1), mz = z * (z_size - 1);
    int dx = std::min(static_cast<int>(floor(mx)), x_size - 2), dy = std::min(static_cast<int>(floor(my)), y_size - 2), dz = std::min(static_cast<int>(floor(mz)), z_size - 2);

    int index[8];
    index[0] = dx * y_size * z_size + dy * z_size + dz;
    index[1] = (dx + 1) * y_size * z_size + dy * z_size + dz;
    index[2] = dx * y_size * z_size + (dy + 1) * z_size + dz;
    index[3] = (dx + 1) * y_size * z_size + (dy + 1) * z_size + dz;
    index[4] = dx * y_size * z_size + dy * z_size + (dz + 1);
    index[5] = (dx + 1) * y_size * z_size + dy * z_size + (dz + 1);
    index[6] = dx * y_size * z_size + (dy + 1) * z_size + (dz + 1);
    index[7] = (dx + 1) * y_size * z_size + (dy + 1) * z_size + (dz + 1);

    for (int i = 0; i < 8; ++i)
    {
        if (index[i] < 0 || static_cast<size_t>(index[i]) >= perlin_cube.size())
        {
            std::cout << "Error: index out of range!\n";
            return;
        }
    }

    // interpolate the noise value via trilinear interpolation
    Real x0 = mx - dx, y0 = my - dy, z0 = mz - dz;
    if (x0 < 0 || x0 >= 1 || y0 < 0 || y0 >= 1 || z0 < 0 || z0 >= 1)
    {
        std::cout << "Error: x, y, z out of range!\n";
        std::cout << mx << ", " << my << ", " << mz << '\n';
        std::cout << "dx: " << dx << ", dy: " << dy << ", dz: " << dz << '\n';
        std::cout << "x0: " << x0 << ", y0: " << y0 << ", z0: " << z0 << '\n';
        return;
    }
    Real coeff[8];
    coeff[0] = (1 - x0) * (1 - y0) * (1 - z0);
    coeff[1] = x0 * (1 - y0) * (1 - z0);
    coeff[2] = (1 - x0) * y0 * (1 - z0);
    coeff[3] = x0 * y0 * (1 - z0);
    coeff[4] = (1 - x0) * (1 - y0) * z0;
    coeff[5] = x0 * (1 - y0) * z0;
    coeff[6] = (1 - x0) * y0 * z0;
    coeff[7] = x0 * y0 * z0;
    noise.reset_zero();
    for (int j = 0; j < 8; ++j)
    {
        noise += coeff[j] * perlin_cube[index[j]];
    }
}

template <typename Real, int FeatureDim>
void Noiser<Real, FeatureDim>::get_random_noise(TinyVector<Real, FeatureDim> &noise)
{
    for (int i = 0; i < FeatureDim; ++i)
        noise[i] = get_random_noise();
}
template <typename Real, int FeatureDim>
Real Noiser<Real, FeatureDim>::get_random_noise()
{
    return ((static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX)) - static_cast<Real>(0.5)) * static_cast<Real>(2);
}

template <typename Real, int FeatureDim>
void Noiser<Real, FeatureDim>::shuffle_perlin_cube()
{
    if (perlin_cube.empty())
        perlin_cube.resize(x_size * y_size * z_size);

    for (auto &noise : perlin_cube)
    {
        for (int j = 0; j < FeatureDim; ++j)
        {
            noise[j] = get_random_noise();
        }
    }
}
/////////////////////////////////////////////////
template class Noiser<double, 3>;