#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <type_traits>

template <typename INT, int Dim, bool SORT = true>
class MySortedTuple
{
public:
	//////////////////////////////////////////////////////////////////////////
	MySortedTuple() noexcept : sorted_vert{} {}

	//////////////////////////////////////////////////////////////////////////
	explicit MySortedTuple(const INT *v) noexcept
	{
		if (v)
		{
			std::copy_n(v, Dim, sorted_vert);
		}
		else
		{
			std::fill_n(sorted_vert, Dim, INT{});
		}
		if constexpr (SORT)
		{
			sort_array();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template <typename... Args,
			  typename = std::enable_if_t<sizeof...(Args) == Dim &&
										  (std::is_convertible_v<Args, INT> && ...)>>
	MySortedTuple(Args &&...args) noexcept
	{
		static_assert(sizeof...(Args) == Dim, "Number of arguments must match Dim");
		INT temp[] = {static_cast<INT>(args)...};
		std::copy_n(temp, Dim, sorted_vert);
		if constexpr (SORT)
		{
			sort_array();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	MySortedTuple(std::initializer_list<INT> init) noexcept
	{
		assert(init.size() <= Dim);
		std::copy(init.begin(), init.end(), sorted_vert);
		std::fill(sorted_vert + init.size(), sorted_vert + Dim, INT{});
		if constexpr (SORT)
		{
			sort_array();
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Rule of 5 with default implementations
	MySortedTuple(const MySortedTuple &) = default;
	MySortedTuple(MySortedTuple &&) = default;
	MySortedTuple &operator=(const MySortedTuple &) = default;
	MySortedTuple &operator=(MySortedTuple &&) = default;
	~MySortedTuple() = default;

	//////////////////////////////////////////////////////////////////////////
	bool operator==(const MySortedTuple &other) const noexcept
	{
		return std::equal(sorted_vert, sorted_vert + Dim, other.sorted_vert);
	}

	//////////////////////////////////////////////////////////////////////////
	bool operator!=(const MySortedTuple &other) const noexcept
	{
		return !(*this == other);
	}

	//////////////////////////////////////////////////////////////////////////
	bool operator<(const MySortedTuple &other) const noexcept
	{
		return std::lexicographical_compare(
			sorted_vert, sorted_vert + Dim,
			other.sorted_vert, other.sorted_vert + Dim);
	}

	//////////////////////////////////////////////////////////////////////////
	bool operator<=(const MySortedTuple &other) const noexcept
	{
		return !(other < *this);
	}

	//////////////////////////////////////////////////////////////////////////
	bool operator>(const MySortedTuple &other) const noexcept
	{
		return other < *this;
	}

	//////////////////////////////////////////////////////////////////////////
	bool operator>=(const MySortedTuple &other) const noexcept
	{
		return !(*this < other);
	}

	//////////////////////////////////////////////////////////////////////////
	INT &operator[](int i) noexcept
	{
		assert(i >= 0 && i < Dim);
		return sorted_vert[i];
	}

	//////////////////////////////////////////////////////////////////////////
	const INT &operator[](int i) const noexcept
	{
		assert(i >= 0 && i < Dim);
		return sorted_vert[i];
	}

	//////////////////////////////////////////////////////////////////////////
	INT *data() noexcept { return sorted_vert; }
	const INT *data() const noexcept { return sorted_vert; }

	//////////////////////////////////////////////////////////////////////////
	constexpr size_t size() const noexcept { return Dim; }

	//////////////////////////////////////////////////////////////////////////
	INT *begin() noexcept { return sorted_vert; }
	const INT *begin() const noexcept { return sorted_vert; }
	INT *end() noexcept { return sorted_vert + Dim; }
	const INT *end() const noexcept { return sorted_vert + Dim; }

private:
	void sort_array() noexcept
	{
		if constexpr (SORT)
		{
			std::sort(sorted_vert, sorted_vert + Dim);
		}
	}

public:
	INT sorted_vert[Dim];
};

namespace std
{
	template <typename INT, int Dim, bool SORT>
	struct hash<MySortedTuple<INT, Dim, SORT>>
	{
		std::size_t operator()(const MySortedTuple<INT, Dim, SORT> &t) const noexcept
		{
			std::size_t seed = 0;
			for (int i = 0; i < Dim; ++i)
			{
				seed ^= std::hash<INT>{}(t.sorted_vert[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};
}
