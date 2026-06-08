#pragma once

#include <sstream>
#include <istream>
#include <ostream>
#include <string>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <cstring>
#include <array>
#include <initializer_list>
#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <stdexcept>

// #define TINYVECTOR_ENABLE_BOUNDS_CHECK

template <typename T, int N>
class TinyVector
{
    static_assert(N > 0, "TinyVector size N must be greater than 0");

public:
    using value_type = T;
    using size_type = int;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    // construction
    constexpr TinyVector() noexcept : data_{} {}
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector(const TinyVector<T, N> &TV) noexcept
    {
        std::copy_n(TV.data_, N, data_);
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector(TinyVector<T, N> &&TV) noexcept
    {
        std::move(TV.data_, TV.data_ + N, data_);
    }
    
    //////////////////////////////////////////////////////////////////////////
    explicit TinyVector(const T *V) noexcept
    {
        if (V) {
            std::copy_n(V, N, data_);
        } else {
            std::fill_n(data_, N, T{});
        }
    }
    
    //////////////////////////////////////////////////////////////////////////
    explicit TinyVector(const char *s)
    {
        std::istringstream ins(s);
        for (int i = 0; i < N; i++)
            ins >> data_[i];
    }
    
    //////////////////////////////////////////////////////////////////////////
    explicit TinyVector(const std::string &s)
    {
        std::istringstream ins(s);
        for (int i = 0; i < N; i++)
            ins >> data_[i];
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector(const std::array<T, N> &TV) noexcept
    {
        std::copy_n(TV.data(), N, data_);
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector(std::initializer_list<T> TV) noexcept
    {
        const auto size = std::min(static_cast<int>(TV.size()), N);
        std::copy_n(TV.begin(), size, data_);
        std::fill(data_ + size, data_ + N, T{});
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N, typename std::enable_if<(M >= 2), int>::type = 0>
    constexpr TinyVector(const T x, const T y) noexcept : data_{x, y} {}
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N, typename std::enable_if<(M >= 3), int>::type = 0>
    constexpr TinyVector(const T x, const T y, const T z) noexcept : data_{x, y, z} {}
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N, typename std::enable_if<(M >= 4), int>::type = 0>
    constexpr TinyVector(const T x, const T y, const T z, const T w) noexcept : data_{x, y, z, w} {}
    
    //////////////////////////////////////////////////////////////////////////
    ~TinyVector() = default;
    
    //////////////////////////////////////////////////////////////////////////
    void reset_zero() noexcept
    {
        std::fill_n(data_, N, T{});
    }
    
    //////////////////////////////////////////////////////////////////////////
    // access
    constexpr operator const T *() const noexcept { return data_; }
    constexpr operator T *() noexcept { return data_; }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr T &operator()(int i) noexcept
    {
#ifdef TINYVECTOR_ENABLE_BOUNDS_CHECK
        if (i < 1 || i > N) {
            throw std::out_of_range("Index out of bounds in TinyVector");
        }
#endif
        return data_[i - 1];
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr const T &operator()(int i) const noexcept
    {
#ifdef TINYVECTOR_ENABLE_BOUNDS_CHECK
        if (i < 1 || i > N) {
            throw std::out_of_range("Index out of bounds in TinyVector");
        }
#endif
        return data_[i - 1];
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr T &operator[](int i) noexcept
    {
#ifdef TINYVECTOR_ENABLE_BOUNDS_CHECK
        if (i < 0 || i >= N) {
            throw std::out_of_range("Index out of bounds in TinyVector");
        }
#endif
        return data_[i];
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr const T &operator[](int i) const noexcept
    {
#ifdef TINYVECTOR_ENABLE_BOUNDS_CHECK
        if (i < 0 || i >= N) {
            throw std::out_of_range("Index out of bounds in TinyVector");
        }
#endif
        return data_[i];
    }
    
    //////////////////////////////////////////////////////////////////////////
    constexpr const T &x() const noexcept { return data_[0]; }
    constexpr T &x() noexcept { return data_[0]; }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    constexpr typename std::enable_if<(M > 1), const T &>::type y() const noexcept
    {
        return data_[1];
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    constexpr typename std::enable_if<(M > 1), T &>::type y() noexcept
    {
        return data_[1];
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    constexpr typename std::enable_if<(M > 2), const T &>::type z() const noexcept
    {
        return data_[2];
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    constexpr typename std::enable_if<(M > 2), T &>::type z() noexcept
    {
        return data_[2];
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    constexpr typename std::enable_if<(M > 3), const T &>::type w() const noexcept
    {
        return data_[3];
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    constexpr typename std::enable_if<(M > 3), T &>::type w() noexcept
    {
        return data_[3];
    }
    
    //////////////////////////////////////////////////////////////////////////
    // assignment
    TinyVector<T, N> &operator=(const TinyVector<T, N> &A) noexcept
    {
        if (this != &A) {
            std::copy_n(A.data_, N, data_);
        }
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator=(TinyVector<T, N> &&A) noexcept
    {
        if (this != &A) {
            std::move(A.data_, A.data_ + N, data_);
        }
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator=(const T *A) noexcept
    {
        if (data_ != A) {
            std::copy_n(A, N, data_);
        }
        return *this;
    }
    
    ////////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator=(const T &scalar) noexcept
    {
        std::fill_n(data_, N, scalar);
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator=(const std::array<T, N> &A) noexcept
    {
        std::copy_n(A.data(), N, data_);
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    // comparison
    bool operator==(const TinyVector<T, N> &rkV) const noexcept
    {
        return std::equal(data_, data_ + N, rkV.data_);
    }
    
    //////////////////////////////////////////////////////////////////////////
    bool operator!=(const TinyVector<T, N> &rkV) const noexcept
    {
        return !(*this == rkV);
    }
    
    //////////////////////////////////////////////////////////////////////////
    bool operator<(const TinyVector<T, N> &rkV) const noexcept
    {
        return std::lexicographical_compare(data_, data_ + N, rkV.data_, rkV.data_ + N);
    }
    
    //////////////////////////////////////////////////////////////////////////
    bool operator<=(const TinyVector<T, N> &rkV) const noexcept
    {
        return !(rkV < *this);
    }
    
    //////////////////////////////////////////////////////////////////////////
    bool operator>(const TinyVector<T, N> &rkV) const noexcept
    {
        return rkV < *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    bool operator>=(const TinyVector<T, N> &rkV) const noexcept
    {
        return !(*this < rkV);
    }

    //////////////////////////////////////////////////////////////////////////
    // arithmetic operations
    TinyVector<T, N> operator+(const TinyVector<T, N> &rkV) const noexcept
    {
        TinyVector<T, N> tmp;
        std::transform(data_, data_ + N, rkV.data_, tmp.data_, std::plus<T>());
        return tmp;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> operator-(const TinyVector<T, N> &rkV) const noexcept
    {
        TinyVector<T, N> tmp;
        std::transform(data_, data_ + N, rkV.data_, tmp.data_, std::minus<T>());
        return tmp;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> operator*(T fScalar) const noexcept
    {
        TinyVector<T, N> tmp;
        std::transform(data_, data_ + N, tmp.data_, 
            [fScalar](const T& val) { return fScalar * val; });
        return tmp;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> operator/(T fScalar) const noexcept
    {
        TinyVector<T, N> tmp;
        if (fScalar != T{}) {
            const T fInvScalar = T{1} / fScalar;
            std::transform(data_, data_ + N, tmp.data_, 
                [fInvScalar](const T& val) { return fInvScalar * val; });
        } else {
            std::fill_n(tmp.data_, N, std::numeric_limits<T>::max());
        }
        return tmp;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> operator-() const noexcept
    {
        TinyVector<T, N> tmp;
        std::transform(data_, data_ + N, tmp.data_, std::negate<T>());
        return tmp;
    }
    
    //////////////////////////////////////////////////////////////////////////
    // arithmetic updates
    TinyVector<T, N> &operator+=(const TinyVector<T, N> &rkV) noexcept
    {
        std::transform(data_, data_ + N, rkV.data_, data_, std::plus<T>());
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator-=(const TinyVector<T, N> &rkV) noexcept
    {
        std::transform(data_, data_ + N, rkV.data_, data_, std::minus<T>());
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator*=(T fScalar) noexcept
    {
        std::transform(data_, data_ + N, data_, 
            [fScalar](const T& val) { return val * fScalar; });
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> &operator/=(T fScalar) noexcept
    {
        if (fScalar != T{}) {
            const T fInvScalar = T{1} / fScalar;
            std::transform(data_, data_ + N, data_, 
                [fInvScalar](const T& val) { return val * fInvScalar; });
        } else {
            std::fill_n(data_, N, std::numeric_limits<T>::max());
        }
        return *this;
    }
    
    //////////////////////////////////////////////////////////////////////////
    // vector operations
    T Length() const noexcept
    {
        return std::sqrt(SquaredLength());
    }
    
    //////////////////////////////////////////////////////////////////////////
    T SquaredLength() const noexcept
    {
        return std::inner_product(data_, data_ + N, data_, T{});
    }
    
    //////////////////////////////////////////////////////////////////////////
    T Dot(const TinyVector<T, N> &TV) const noexcept
    {
        return std::inner_product(data_, data_ + N, TV.data_, T{});
    }
    
    //////////////////////////////////////////////////////////////////////////
    template <int M = N>
    typename std::enable_if<(M >= 2), TinyVector<T, N>>::type Cross(const TinyVector<T, N> &TV) const noexcept
    {
        TinyVector<T, N> tmp;

        if constexpr (N == 2) {
            tmp[0] = data_[0] * TV.data_[1] - data_[1] * TV.data_[0];
            if constexpr (N > 1) tmp[1] = T{};
        }
        else if constexpr (N == 3) {
            tmp[0] = data_[1] * TV.data_[2] - data_[2] * TV.data_[1];
            tmp[1] = data_[2] * TV.data_[0] - data_[0] * TV.data_[2];
            tmp[2] = data_[0] * TV.data_[1] - data_[1] * TV.data_[0];
        }
        else if constexpr (N > 3) {
            for (int i = 0; i < N; i++) {
                const int id1 = (i + 1) % N;
                const int id2 = (i + 2) % N;
                tmp[i] = data_[id1] * TV.data_[id2] - data_[id2] * TV.data_[id1];
            }
        }
        return tmp;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> UnitCross(const TinyVector<T, N> &TV) const noexcept
    {
        return Cross(TV).GetNormalized();
    }
    
    //////////////////////////////////////////////////////////////////////////
    T max() const noexcept
    {
        return *std::max_element(data_, data_ + N);
    }
    
    size_t max_index() const noexcept
    {
        return std::distance(data_, std::max_element(data_, data_ + N));
    }
    
    //////////////////////////////////////////////////////////////////////////
    T min() const noexcept
    {
        return *std::min_element(data_, data_ + N);
    }
    
    size_t min_index() const noexcept
    {
        return std::distance(data_, std::min_element(data_, data_ + N));
    }
    
    //////////////////////////////////////////////////////////////////////////
    T sum() const noexcept
    {
        return std::accumulate(data_, data_ + N, T{});
    }

    //////////////////////////////////////////////////////////////////////////
    T Normalize() noexcept
    {
        const T fLength = Length();
        if (fLength != T{}) {
            const T fInvLength = T{1} / fLength;
            std::transform(data_, data_ + N, data_, 
                [fInvLength](const T& val) { return val * fInvLength; });
        } else {
            std::fill_n(data_, N, T{});
        }
        return fLength;
    }
    
    //////////////////////////////////////////////////////////////////////////
    TinyVector<T, N> GetNormalized() const noexcept
    {
        const T fLength = Length();
        TinyVector<T, N> tmp;
        
        if (fLength != T{}) {
            const T fInvLength = T{1} / fLength;
            std::transform(data_, data_ + N, tmp.data_, 
                [fInvLength](const T& val) { return val * fInvLength; });
        } else {
            std::fill_n(tmp.data_, N, T{});
        }
        return tmp;
    }

    //////////////////////////////////////////////////////////////////////////
    static void ComputeExtremes(int iVQuantity, const TinyVector<T, N> *akPoint,
                                TinyVector<T, N> &rkMin, TinyVector<T, N> &rkMax) noexcept
    {
        assert(iVQuantity > 0 && akPoint);

        rkMin = akPoint[0];
        rkMax = rkMin;
        
        for (int i = 1; i < iVQuantity; i++) {
            const TinyVector<T, N> &rkPoint = akPoint[i];
            std::transform(rkMin.data_, rkMin.data_ + N, rkPoint.data_, rkMin.data_,
                [](const T& a, const T& b) { return std::min(a, b); });
            std::transform(rkMax.data_, rkMax.data_ + N, rkPoint.data_, rkMax.data_,
                [](const T& a, const T& b) { return std::max(a, b); });
        }
    }
    
    //////////////////////////////////////////////////////////////////////////
    void randomize(const T start = T{}, const T end = T{1}) noexcept
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(start, end);

        std::generate_n(data_, N, [&dis](void) -> T {
            static std::random_device rd_local;
            static std::mt19937 gen_local(rd_local());
            return dis(gen_local);
        });
    }

    //////////////////////////////////////////////////////////////////////////
    // STL compatibility
    constexpr size_type size() const noexcept { return N; }
    constexpr bool empty() const noexcept { return N == 0; }
    constexpr iterator begin() noexcept { return data_; }
    constexpr const_iterator begin() const noexcept { return data_; }
    constexpr iterator end() noexcept { return data_ + N; }
    constexpr const_iterator end() const noexcept { return data_ + N; }
    constexpr pointer data() noexcept { return data_; }
    constexpr const_pointer data() const noexcept { return data_; }

private:
    T data_[N];
};

template <typename T, int N>
TinyVector<T, N> operator*(T fScalar, const TinyVector<T, N> &rkV) noexcept
{
    return rkV * fScalar;
}

template <typename T, int N>
T operator*(const TinyVector<T, N> &rkU, const TinyVector<T, N> &rkV) noexcept
{
    return rkU.Dot(rkV);
}

template <typename T, int N>
std::ostream &operator<<(std::ostream &s, const TinyVector<T, N> &A)
{
    for (int i = 0; i < N - 1; i++) {
        s << A[i] << " ";
    }
    s << A[N - 1];
    return s;
}

template <typename T, int N>
std::istream &operator>>(std::istream &s, TinyVector<T, N> &A)
{
    for (int i = 0; i < N; i++) {
        s >> A[i];
    }
    return s;
}

template <typename T>
T dot(int N, const T *vec_x, const T *vec_y) noexcept
{
    return std::inner_product(vec_x, vec_x + N, vec_y, T{});
}

template <typename Real>
void GenerateComplementBasis(TinyVector<Real, 3> &u, TinyVector<Real, 3> &v, const TinyVector<Real, 3> &w) noexcept
{
    Real invLength;

    if (std::abs(w[0]) >= std::abs(w[1])) {
        invLength = Real{1} / std::sqrt(w[0] * w[0] + w[2] * w[2]);
        u[0] = -w[2] * invLength;
        u[1] = Real{};
        u[2] = w[0] * invLength;
        v[0] = w[1] * u[2];
        v[1] = w[2] * u[0] - w[0] * u[2];
        v[2] = -w[1] * u[0];
    } else {
        invLength = Real{1} / std::sqrt(w[1] * w[1] + w[2] * w[2]);
        u[0] = Real{};
        u[1] = w[2] * invLength;
        u[2] = -w[1] * invLength;
        v[0] = w[1] * u[2] - w[2] * u[1];
        v[1] = -w[0] * u[2];
        v[2] = w[0] * u[1];
    }
}
