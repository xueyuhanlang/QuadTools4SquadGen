#pragma once

#include <cmath>
#include <algorithm>

// https://github.com/brainexcerpts/3x3_polar_decomposition
// -----------------------------------------------------------------------------

template <typename Real>
class Eigen3x3
{
public:
    Eigen3x3(const Real A[3][3], Real V[3][3], Real d[3])
    {
        Real e[3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                V[i][j] = A[i][j];
            }
        }
        tred2(V, d, e);
        tql2(V, d, e);
    }

protected:
    Real hypot2(Real x, Real y)
    {
        return sqrt(x * x + y * y);
    }
    // Symmetric Householder reduction to tridiagonal form.
    void tred2(Real V[3][3], Real d[3], Real e[3])
    {

        //  This is derived from the Algol procedures tred2 by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        for (int j = 0; j < 3; j++)
        {
            d[j] = V[3 - 1][j];
        }

        // Householder reduction to tridiagonal form.

        for (int i = 3 - 1; i > 0; i--)
        {

            // Scale to avoid under/overflow.

            Real scale = 0.0;
            Real h = 0.0;
            for (int k = 0; k < i; k++)
            {
                scale = scale + fabs(d[k]);
            }
            if (scale == 0.0)
            {
                e[i] = d[i - 1];
                for (int j = 0; j < i; j++)
                {
                    d[j] = V[i - 1][j];
                    V[i][j] = 0.0;
                    V[j][i] = 0.0;
                }
            }
            else
            {

                // Generate Householder vector.

                for (int k = 0; k < i; k++)
                {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                Real f = d[i - 1];
                Real g = sqrt(h);
                if (f > 0)
                {
                    g = -g;
                }
                e[i] = scale * g;
                h = h - f * g;
                d[i - 1] = f - g;
                for (int j = 0; j < i; j++)
                {
                    e[j] = 0.0;
                }

                // Apply similarity transformation to remaining columns.

                for (int j = 0; j < i; j++)
                {
                    f = d[j];
                    V[j][i] = f;
                    g = e[j] + V[j][j] * f;
                    for (int k = j + 1; k <= i - 1; k++)
                    {
                        g += V[k][j] * d[k];
                        e[k] += V[k][j] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for (int j = 0; j < i; j++)
                {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                Real hh = f / (h + h);
                for (int j = 0; j < i; j++)
                {
                    e[j] -= hh * d[j];
                }
                for (int j = 0; j < i; j++)
                {
                    f = d[j];
                    g = e[j];
                    for (int k = j; k <= i - 1; k++)
                    {
                        V[k][j] -= (f * e[k] + g * d[k]);
                    }
                    d[j] = V[i - 1][j];
                    V[i][j] = 0.0;
                }
            }
            d[i] = h;
        }

        // Accumulate transformations.

        for (int i = 0; i < 3 - 1; i++)
        {
            V[3 - 1][i] = V[i][i];
            V[i][i] = 1.0;
            Real h = d[i + 1];
            if (h != 0.0)
            {
                for (int k = 0; k <= i; k++)
                {
                    d[k] = V[k][i + 1] / h;
                }
                for (int j = 0; j <= i; j++)
                {
                    Real g = 0.0;
                    for (int k = 0; k <= i; k++)
                    {
                        g += V[k][i + 1] * V[k][j];
                    }
                    for (int k = 0; k <= i; k++)
                    {
                        V[k][j] -= g * d[k];
                    }
                }
            }
            for (int k = 0; k <= i; k++)
            {
                V[k][i + 1] = 0.0;
            }
        }
        for (int j = 0; j < 3; j++)
        {
            d[j] = V[3 - 1][j];
            V[3 - 1][j] = 0.0;
        }
        V[3 - 1][3 - 1] = 1.0;
        e[0] = 0.0;
    }
    // Symmetric tridiagonal QL algorithm.
    void tql2(Real V[3][3], Real d[3], Real e[3])
    {

        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        for (int i = 1; i < 3; i++)
        {
            e[i - 1] = e[i];
        }
        e[3 - 1] = 0.0;

        Real f = 0.0f;
        Real tst1 = 0.0f;
        Real eps = (Real)pow(2.0, -52.0);
        for (int l = 0; l < 3; l++)
        {

            // Find small subdiagonal element

            tst1 = std::max(tst1, fabs(d[l]) + fabs(e[l]));
            int m = l;
            while (m < 3)
            {
                if (fabs(e[m]) <= eps * tst1)
                {
                    break;
                }
                m++;
            }

            // If m == l, d[l] is an eigenvalue,
            // otherwise, iterate.

            if (m > l)
            {
                int iter = 0;
                do
                {
                    iter = iter + 1; // (Could check iteration count here.)

                    // Compute implicit shift

                    Real g = d[l];
                    Real p = (d[l + 1] - g) / (2 * e[l]);
                    Real r = hypot2(p, 1);
                    if (p < 0)
                    {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    Real dl1 = d[l + 1];
                    Real h = g - d[l];
                    for (int i = l + 2; i < 3; i++)
                    {
                        d[i] -= h;
                    }
                    f = f + h;

                    // Implicit QL transformation.

                    p = d[m];
                    Real c = 1.0;
                    Real c2 = c;
                    Real c3 = c;
                    Real el1 = e[l + 1];
                    Real s = 0.0;
                    Real s2 = 0.0;
                    for (int i = m - 1; i >= l; i--)
                    {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = hypot2(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        // Accumulate transformation.

                        for (int k = 0; k < 3; k++)
                        {
                            h = V[k][i + 1];
                            V[k][i + 1] = s * V[k][i] + c * h;
                            V[k][i] = c * V[k][i] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;

                    // Check for convergence.

                } while (fabs(e[l]) > eps * tst1);
            }
            d[l] = d[l] + f;
            e[l] = 0.0;
        }

        // Sort eigenvalues and corresponding vectors.

        for (int i = 0; i < 3 - 1; i++)
        {
            int k = i;
            Real p = d[i];
            for (int j = i + 1; j < 3; j++)
            {
                if (d[j] < p)
                {
                    k = j;
                    p = d[j];
                }
            }
            if (k != i)
            {
                d[k] = d[i];
                d[i] = p;
                for (int j = 0; j < 3; j++)
                {
                    p = V[j][i];
                    V[j][i] = V[j][k];
                    V[j][k] = p;
                }
            }
        }
    }
};
