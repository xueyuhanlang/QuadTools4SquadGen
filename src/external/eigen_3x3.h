#pragma once

#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

// https://github.com/brainexcerpts/3x3_polar_decomposition
// Computes eigenvalues (d) and eigenvectors (columns of V) of a real symmetric 3x3 matrix A.
// Eigenvalues are returned in ascending order.

template <typename Real>
class Eigen3x3
{
public:
    static constexpr int N = 3;
    using Matrix = std::array<std::array<Real, N>, N>;
    using Vector = std::array<Real, N>;

    // C-style array interface for backward compatibility.
    // On return, d_out[i] are eigenvalues (ascending), columns of V_out are eigenvectors.
    Eigen3x3(const Real A[N][N], Real V_out[N][N], Real d_out[N]) noexcept
    {
        Matrix V{};
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                V[i][j] = A[i][j];

        Vector d{}, e{};
        tred2(V, d, e);
        tql2(V, d, e);

        for (int i = 0; i < N; ++i) {
            d_out[i] = d[i];
            for (int j = 0; j < N; ++j)
                V_out[i][j] = V[i][j];
        }
    }

    // Modern std::array interface.
    Eigen3x3(const Matrix& A, Matrix& V, Vector& d) noexcept
    {
        V = A;
        Vector e{};
        tred2(V, d, e);
        tql2(V, d, e);
    }

private:
    // Symmetric Householder reduction to tridiagonal form.
    // Derived from Algol procedures tred2 by Bowdler, Martin, Reinsch, and Wilkinson,
    // Handbook for Auto. Comp., Vol.ii-Linear Algebra (EISPACK).
    static void tred2(Matrix& V, Vector& d, Vector& e) noexcept
    {
        for (int j = 0; j < N; ++j)
            d[j] = V[N - 1][j];

        for (int i = N - 1; i > 0; --i) {
            Real scale = Real(0);
            Real h = Real(0);
            for (int k = 0; k < i; ++k)
                scale += std::fabs(d[k]);

            if (scale == Real(0)) {
                e[i] = d[i - 1];
                for (int j = 0; j < i; ++j) {
                    d[j]    = V[i - 1][j];
                    V[i][j] = Real(0);
                    V[j][i] = Real(0);
                }
            } else {
                for (int k = 0; k < i; ++k) {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                Real f = d[i - 1];
                Real g = std::sqrt(h);
                if (f > Real(0))
                    g = -g;
                e[i]     = scale * g;
                h       -= f * g;
                d[i - 1] = f - g;
                for (int j = 0; j < i; ++j)
                    e[j] = Real(0);

                // Apply similarity transformation to remaining columns.
                for (int j = 0; j < i; ++j) {
                    f        = d[j];
                    V[j][i]  = f;
                    g        = e[j] + V[j][j] * f;
                    for (int k = j + 1; k <= i - 1; ++k) {
                        g    += V[k][j] * d[k];
                        e[k] += V[k][j] * f;
                    }
                    e[j] = g;
                }
                f = Real(0);
                for (int j = 0; j < i; ++j) {
                    e[j] /= h;
                    f    += e[j] * d[j];
                }
                const Real hh = f / (h + h);
                for (int j = 0; j < i; ++j)
                    e[j] -= hh * d[j];
                for (int j = 0; j < i; ++j) {
                    f = d[j];
                    g = e[j];
                    for (int k = j; k <= i - 1; ++k)
                        V[k][j] -= f * e[k] + g * d[k];
                    d[j]    = V[i - 1][j];
                    V[i][j] = Real(0);
                }
            }
            d[i] = h;
        }

        // Accumulate transformations.
        for (int i = 0; i < N - 1; ++i) {
            V[N - 1][i] = V[i][i];
            V[i][i]     = Real(1);
            const Real h = d[i + 1];
            if (h != Real(0)) {
                for (int k = 0; k <= i; ++k)
                    d[k] = V[k][i + 1] / h;
                for (int j = 0; j <= i; ++j) {
                    Real g = Real(0);
                    for (int k = 0; k <= i; ++k)
                        g += V[k][i + 1] * V[k][j];
                    for (int k = 0; k <= i; ++k)
                        V[k][j] -= g * d[k];
                }
            }
            for (int k = 0; k <= i; ++k)
                V[k][i + 1] = Real(0);
        }
        for (int j = 0; j < N; ++j) {
            d[j]            = V[N - 1][j];
            V[N - 1][j]     = Real(0);
        }
        V[N - 1][N - 1] = Real(1);
        e[0]            = Real(0);
    }

    // Symmetric tridiagonal QL algorithm.
    // Derived from Algol procedures tql2 by Bowdler, Martin, Reinsch, and Wilkinson,
    // Handbook for Auto. Comp., Vol.ii-Linear Algebra (EISPACK).
    static void tql2(Matrix& V, Vector& d, Vector& e) noexcept
    {
        for (int i = 1; i < N; ++i)
            e[i - 1] = e[i];
        e[N - 1] = Real(0);

        Real f    = Real(0);
        Real tst1 = Real(0);
        const Real eps = std::numeric_limits<Real>::epsilon();

        for (int l = 0; l < N; ++l) {
            tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
            int m = l;
            while (m < N) {
                if (std::fabs(e[m]) <= eps * tst1)
                    break;
                ++m;
            }

            if (m > l) {
                do {
                    Real g = d[l];
                    Real p = (d[l + 1] - g) / (Real(2) * e[l]);
                    Real r = std::hypot(p, Real(1));
                    if (p < Real(0))
                        r = -r;
                    d[l]       = e[l] / (p + r);
                    d[l + 1]   = e[l] * (p + r);
                    const Real dl1 = d[l + 1];
                    Real h     = g - d[l];
                    for (int i = l + 2; i < N; ++i)
                        d[i] -= h;
                    f += h;

                    // Implicit QL transformation.
                    p              = d[m];
                    Real c         = Real(1);
                    Real c2        = Real(1);
                    Real c3        = Real(1);
                    const Real el1 = e[l + 1];
                    Real s         = Real(0);
                    Real s2        = Real(0);

                    for (int i = m - 1; i >= l; --i) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g  = c * e[i];
                        h  = c * p;
                        r  = std::hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        // Accumulate transformation.
                        for (int k = 0; k < N; ++k) {
                            h            = V[k][i + 1];
                            V[k][i + 1]  = s * V[k][i] + c * h;
                            V[k][i]      = c * V[k][i] - s * h;
                        }
                    }
                    p    = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                } while (std::fabs(e[l]) > eps * tst1);
            }
            d[l] += f;
            e[l]  = Real(0);
        }

        // Sort eigenvalues and corresponding vectors in ascending order.
        for (int i = 0; i < N - 1; ++i) {
            int  k = i;
            Real p = d[i];
            for (int j = i + 1; j < N; ++j) {
                if (d[j] < p) {
                    k = j;
                    p = d[j];
                }
            }
            if (k != i) {
                d[k] = d[i];
                d[i] = p;
                for (int j = 0; j < N; ++j)
                    std::swap(V[j][i], V[j][k]);
            }
        }
    }
};
