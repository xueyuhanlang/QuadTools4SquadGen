#pragma once
#include <vector>
#include <unordered_set>
#include <cstdlib>

void greedy_graph_coloring(const size_t num_vertices, const std::vector<std::unordered_set<size_t>>& edges, std::vector<std::vector<size_t>>& colored_vertices);