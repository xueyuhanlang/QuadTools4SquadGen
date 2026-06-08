#ifndef TINY_MATRIX_H
#define TINY_MATRIX_H

#include "TinyVector.h"
#include <array>
#include <numeric>
#include <algorithm>
#include <limits>

template <typename T>
class ColumnMatrix3
{
public:
    using value_type = T;
    using column_type = TinyVector<T, 3>;
    using size_type = int;
    
private:
    std::array<column_type, 3> V_;

public:
    //////////////////////////////////////////////////////////////////////////
    constexpr ColumnMatrix3() noexcept = default;
    
    //////////////////////////////////////////////////////////////////////////
    constexpr ColumnMatrix3(const column_type& T0, const column_type& T1, const column_type& T2) noexcept
        : V_{T0, T1, T2} {}
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3(const T* T0, const T* T1, const T* T2) noexcept
        : V_{column_type(T0), column_type(T1), column_type(T2)} {}
    
    //////////////////////////////////////////////////////////////////////////
    explicit ColumnMatrix3(const T* V) noexcept
        : V_{column_type(&V[0]), column_type(&V[3]), column_type(&V[6])} {}
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3(const ColumnMatrix3& CM) = default;
    ColumnMatrix3(ColumnMatrix3&& CM) = default;
    ColumnMatrix3& operator=(const ColumnMatrix3& CM) = default;
    ColumnMatrix3& operator=(ColumnMatrix3&& CM) = default;
    ~ColumnMatrix3() = default;
    
    //////////////////////////////////////////////////////////////////////////
    constexpr ColumnMatrix3(T a0, T a1, T a2, T a3, T a4, T a5,
                           T a6, T a7, T a8, bool columnMajor = true) noexcept
    {
        SetEntries(a0, a1, a2, a3, a4, a5, a6, a7, a8, columnMajor);
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr void SetEntries(T a0, T a1, T a2, T a3, T a4, T a5,
                             T a6, T a7, T a8, bool columnMajor = true) noexcept
    {
        if (columnMajor) {
            V_[0] = column_type(a0, a1, a2);
            V_[1] = column_type(a3, a4, a5);
            V_[2] = column_type(a6, a7, a8);
        } else {
            V_[0] = column_type(a0, a3, a6);
            V_[1] = column_type(a1, a4, a7);
            V_[2] = column_type(a2, a5, a8);
        }
    }
    
    //////////////////////////////////////////////////////////////////////////
    column_type operator*(const column_type& R) const noexcept
    {
        return column_type(
            V_[0][0] * R[0] + V_[1][0] * R[1] + V_[2][0] * R[2],
            V_[0][1] * R[0] + V_[1][1] * R[1] + V_[2][1] * R[2],
            V_[0][2] * R[0] + V_[1][2] * R[1] + V_[2][2] * R[2]
        );
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 operator*(const ColumnMatrix3& R) const noexcept
    {
        ColumnMatrix3 result;
        
        // Optimized matrix multiplication using cache-friendly access pattern
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                result.V_[col][row] = V_[0][row] * R.V_[col][0] + 
                                     V_[1][row] * R.V_[col][1] + 
                                     V_[2][row] * R.V_[col][2];
            }
        }
        return result;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 TransposeTimes(const ColumnMatrix3& R) const noexcept
    {
        return ColumnMatrix3(
            V_[0].Dot(R.V_[0]), V_[1].Dot(R.V_[0]), V_[2].Dot(R.V_[0]),
            V_[0].Dot(R.V_[1]), V_[1].Dot(R.V_[1]), V_[2].Dot(R.V_[1]),
            V_[0].Dot(R.V_[2]), V_[1].Dot(R.V_[2]), V_[2].Dot(R.V_[2])
        );
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 TimesTranspose(const ColumnMatrix3& R) const noexcept
    {
        ColumnMatrix3 result;
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                result.V_[col][row] = V_[0][row] * R.V_[0][col] + 
                                     V_[1][row] * R.V_[1][col] + 
                                     V_[2][row] * R.V_[2][col];
            }
        }
        return result;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 TransposeTimesTranspose(const ColumnMatrix3& R) const noexcept
    {
        ColumnMatrix3 result;
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                result.V_[col][row] = R.V_[0][col] * V_[row][0] + 
                                     R.V_[1][col] * V_[row][1] + 
                                     R.V_[2][col] * V_[row][2];
            }
        }
        return result;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 TimesDiagonal(const column_type& R) const noexcept
    {
        return ColumnMatrix3(R[0] * V_[0], R[1] * V_[1], R[2] * V_[2]);
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 DiagonalTimes(const column_type& R) const noexcept
    {
        return ColumnMatrix3(
            R[0] * V_[0][0], R[1] * V_[0][1], R[2] * V_[0][2],
            R[0] * V_[1][0], R[1] * V_[1][1], R[2] * V_[1][2],
            R[0] * V_[2][0], R[1] * V_[2][1], R[2] * V_[2][2]
        );
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr const column_type& operator[](int i) const noexcept
    {
        return V_[i];
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr column_type& operator[](int i) noexcept
    {
        return V_[i];
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 Inverse() const noexcept
    {
        ColumnMatrix3 MI;
        
        // Calculate cofactor matrix using explicit formulas for 3x3
        MI.V_[0][0] = V_[1][1] * V_[2][2] - V_[2][1] * V_[1][2];
        MI.V_[0][1] = V_[2][1] * V_[0][2] - V_[0][1] * V_[2][2];
        MI.V_[0][2] = V_[0][1] * V_[1][2] - V_[1][1] * V_[0][2];
        
        MI.V_[1][0] = V_[2][0] * V_[1][2] - V_[1][0] * V_[2][2];
        MI.V_[1][1] = V_[0][0] * V_[2][2] - V_[2][0] * V_[0][2];
        MI.V_[1][2] = V_[1][0] * V_[0][2] - V_[0][0] * V_[1][2];
        
        MI.V_[2][0] = V_[1][0] * V_[2][1] - V_[2][0] * V_[1][1];
        MI.V_[2][1] = V_[2][0] * V_[0][1] - V_[0][0] * V_[2][1];
        MI.V_[2][2] = V_[0][0] * V_[1][1] - V_[1][0] * V_[0][1];
        
        // Calculate determinant
        const T det = V_[0][0] * MI.V_[0][0] + V_[1][0] * MI.V_[0][1] + V_[2][0] * MI.V_[0][2];
        
        // Check for singular matrix using appropriate epsilon
        constexpr T epsilon = std::numeric_limits<T>::epsilon() * T{100};
        if (std::abs(det) > epsilon) {
            const T inv_det = T{1} / det;
            MI.V_[0] *= inv_det;
            MI.V_[1] *= inv_det;
            MI.V_[2] *= inv_det;
        }
        return MI;
    }
    
    //////////////////////////////////////////////////////////////////////////
    T Determinant() const noexcept
    {
        return V_[0][0] * (V_[1][1] * V_[2][2] - V_[2][1] * V_[1][2]) + 
               V_[1][0] * (V_[2][1] * V_[0][2] - V_[0][1] * V_[2][2]) + 
               V_[2][0] * (V_[0][1] * V_[1][2] - V_[1][1] * V_[0][2]);
    }
    
    ////////////////////////////////////////////////////////////////////////
    ColumnMatrix3& MakeZero() noexcept
    {
        std::for_each(V_.begin(), V_.end(), [](column_type& v) { v.reset_zero(); });
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3& MakeIdentity() noexcept
    {
        V_[0] = column_type(T{1}, T{0}, T{0});
        V_[1] = column_type(T{0}, T{1}, T{0});
        V_[2] = column_type(T{0}, T{0}, T{1});
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    // Arithmetic operations using TinyVector operations
    ColumnMatrix3 operator+(const ColumnMatrix3& mat) const noexcept
    {
        return ColumnMatrix3(V_[0] + mat.V_[0], V_[1] + mat.V_[1], V_[2] + mat.V_[2]);
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 operator-(const ColumnMatrix3& mat) const noexcept
    {
        return ColumnMatrix3(V_[0] - mat.V_[0], V_[1] - mat.V_[1], V_[2] - mat.V_[2]);
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 operator*(T scalar) const noexcept
    {
        return ColumnMatrix3(scalar * V_[0], scalar * V_[1], scalar * V_[2]);
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 operator/(T scalar) const noexcept
    {
        if (scalar != T{}) {
            const T inv_scalar = T{1} / scalar;
            return ColumnMatrix3(inv_scalar * V_[0], inv_scalar * V_[1], inv_scalar * V_[2]);
        }
        return ColumnMatrix3{};
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 operator-() const noexcept
    {
        return ColumnMatrix3(-V_[0], -V_[1], -V_[2]);
    }
    
    //////////////////////////////////////////////////////////////////////////
    // In-place arithmetic operations
    ColumnMatrix3& operator+=(const ColumnMatrix3& mat) noexcept
    {
        V_[0] += mat.V_[0];
        V_[1] += mat.V_[1];
        V_[2] += mat.V_[2];
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3& operator-=(const ColumnMatrix3& mat) noexcept
    {
        V_[0] -= mat.V_[0];
        V_[1] -= mat.V_[1];
        V_[2] -= mat.V_[2];
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3& operator*=(T scalar) noexcept
    {
        V_[0] *= scalar;
        V_[1] *= scalar;
        V_[2] *= scalar;
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3& operator/=(T scalar) noexcept
    {
        if (scalar != T{}) {
            V_[0] /= scalar;
            V_[1] /= scalar;
            V_[2] /= scalar;
        }
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    ColumnMatrix3 Transpose() const noexcept
    {
        return ColumnMatrix3(
            V_[0][0], V_[0][1], V_[0][2], 
            V_[1][0], V_[1][1], V_[1][2], 
            V_[2][0], V_[2][1], V_[2][2], 
            false
        );
    }
    
    //////////////////////////////////////////////////////////////////////////
    // Additional utility methods
    constexpr T* data() noexcept { return V_[0].data(); }
    constexpr const T* data() const noexcept { return V_[0].data(); }
    
    constexpr size_type size() const noexcept { return 9; }
    constexpr size_type rows() const noexcept { return 3; }
    constexpr size_type cols() const noexcept { return 3; }
    
    // STL-style iterators
    constexpr auto begin() noexcept { return V_.begin(); }
    constexpr auto begin() const noexcept { return V_.begin(); }
    constexpr auto end() noexcept { return V_.end(); }
    constexpr auto end() const noexcept { return V_.end(); }
    
    // Matrix norms and properties
    T FrobeniusNorm() const noexcept
    {
        return std::sqrt(V_[0].SquaredLength() + V_[1].SquaredLength() + V_[2].SquaredLength());
    }
    
    T FrobeniusNormSquared() const noexcept
    {
        return V_[0].SquaredLength() + V_[1].SquaredLength() + V_[2].SquaredLength();
    }
    
    T Trace() const noexcept
    {
        return V_[0][0] + V_[1][1] + V_[2][2];
    }
    
    // Check if matrix is symmetric within tolerance
    bool IsSymmetric(T tolerance = std::numeric_limits<T>::epsilon() * T{100}) const noexcept
    {
        return std::abs(V_[0][1] - V_[1][0]) <= tolerance &&
               std::abs(V_[0][2] - V_[2][0]) <= tolerance &&
               std::abs(V_[1][2] - V_[2][1]) <= tolerance;
    }
    
    // Check if matrix is orthogonal
    bool IsOrthogonal(T tolerance = std::numeric_limits<T>::epsilon() * T{100}) const noexcept
    {
        auto should_be_identity = TransposeTimes(*this);
        return std::abs(should_be_identity.V_[0][0] - T{1}) <= tolerance &&
               std::abs(should_be_identity.V_[1][1] - T{1}) <= tolerance &&
               std::abs(should_be_identity.V_[2][2] - T{1}) <= tolerance &&
               std::abs(should_be_identity.V_[0][1]) <= tolerance &&
               std::abs(should_be_identity.V_[0][2]) <= tolerance &&
               std::abs(should_be_identity.V_[1][2]) <= tolerance;
    }
    
    // Get row vector
    column_type GetRow(int i) const noexcept
    {
        return column_type(V_[0][i], V_[1][i], V_[2][i]);
    }
    
    // Set row vector
    void SetRow(int i, const column_type& row) noexcept
    {
        V_[0][i] = row[0];
        V_[1][i] = row[1];
        V_[2][i] = row[2];
    }
};

// Non-member operators
template <typename T>
constexpr ColumnMatrix3<T> operator*(T scalar, const ColumnMatrix3<T>& mat) noexcept
{
    return mat * scalar;
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const ColumnMatrix3<T>& A)
{
    s << A[0][0] << ' ' << A[1][0] << ' ' << A[2][0] << '\n';
    s << A[0][1] << ' ' << A[1][1] << ' ' << A[2][1] << '\n';
    s << A[0][2] << ' ' << A[1][2] << ' ' << A[2][2] << '\n';
    return s;
}

// Comparison operators
template <typename T>
bool operator==(const ColumnMatrix3<T>& lhs, const ColumnMatrix3<T>& rhs) noexcept
{
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
}

template <typename T>
bool operator!=(const ColumnMatrix3<T>& lhs, const ColumnMatrix3<T>& rhs) noexcept
{
    return !(lhs == rhs);
}

// Type aliases using modern C++
using ColumnMatrix3d = ColumnMatrix3<double>;
using ColumnMatrix3f = ColumnMatrix3<float>;
using ColumnMatrix3i = ColumnMatrix3<int>;

#endif
