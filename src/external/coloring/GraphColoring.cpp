#include "GraphColoring.h"
#include <algorithm>

void greedy_graph_coloring(const size_t num_vertices, const std::vector<std::unordered_set<size_t>> &edges, std::vector<std::vector<size_t>> &colored_vertices)
{
	std::vector<int> result(num_vertices, -1);
	result[0] = 0;
	std::vector<bool> available(num_vertices, false);

	int max_color = 0;
	for (size_t u = 1; u < num_vertices; u++)
	{
		for (size_t v : edges[u])
		{
			if (result[v] != -1)
				available[result[v]] = true;
		}
		int cr;
		for (cr = 0; cr < (int)num_vertices; cr++)
		{
			if (available[cr] == false)
				break;
		}

		result[u] = cr;
		max_color = std::max(cr, max_color);

		for (size_t v : edges[u])
		{
			if (result[v] != -1)
				available[result[v]] = false;
		}
	}
	colored_vertices.resize(max_color + 1);
	for (size_t i = 0; i < num_vertices; i++)
	{
		colored_vertices[result[i]].emplace_back(i);
	}
}