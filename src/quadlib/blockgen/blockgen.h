#pragma once

#include "Mesh3D.h"
#include <unordered_set>
#include "MyTuple.h"

template <typename Real>
class BlockGen
{
public:
	BlockGen(int _block_num = 8, int _resolution = 8, bool _use_ccsubdiv = true, int _subdiv_level = 2,
			 float _perturb_level = 0.2f, float _scale_range = 1.0f, int _seed = 0);
	~BlockGen();

	void export_ply(const std::string &filename);
	void export_obj(const std::string &filename);
	int get_genus();
	////////////////////////////////////////////////
protected:
	void insert_empty_block(int x, int y, int z,
							const std::vector<bool> &gridtag,
							const std::unordered_set<MySortedTuple<int, 3, false>> &empty_blocks_set, std::unordered_set<MySortedTuple<int, 3, false>> &added_empty_blocks_set);
	////////////////////////////////////////////////
	void remove_block_region(int x, int y, int z,
							 std::vector<bool> &gridtag,
							 std::vector<MySortedTuple<int, 3, false>> &empty_blocks,
							 std::unordered_set<MySortedTuple<int, 3, false>> &empty_blocks_set);
	/////////////////////////////////////////////////
	bool range_check(int x, int y, int z);
	bool one_ring_check(int x, int y, int z, int nx, int ny, int nz);
	bool not_same_block(int x, int y, int z, int nx, int ny, int nz);
	/////////////////////////////////////////////////
	bool check_cells(bool use_block, // true for non empty block, false for empty block
					 std::vector<bool> &gridtag,
					 std::vector<MySortedTuple<int, 3, false>> &nonempty_blocks,
					 std::vector<MySortedTuple<int, 3, false>> &empty_blocks,
					 std::unordered_set<MySortedTuple<int, 3, false>> &empty_blocks_set);
	/////////////////////////////////////////////////
	bool check_block_removability(const MySortedTuple<int, 3, false> &block,
								  std::vector<bool> &gridtag);
	/////////////////////////////////////////////////
	bool check_block_removability(const int x, const int y, const int z,
								  std::vector<bool> &gridtag);
	/////////////////////////////////////////////////
	bool next_cell(int x, int y, int z, int axis, int dir, int &nx, int &ny, int &nz, std::vector<bool> &gridtag);
	/////////////////////////////////////////////////
	bool is_interior_block(int x, int y, int z, const std::vector<bool> &gridtag);
	/////////////////////////////////////////////////
	void choose_axis_and_direction(int x, int y, int z, int &axis, int &direction, const std::vector<bool> &gridtag);
	/////////////////////////////////////////////////
	MeshLib::Mesh3D<Real> *create_block_mesh();
	//////////////////////////////////////////////////////////////////////////
	MeshLib::Mesh3D<Real> *pick_max_mesh_compoent(MeshLib::Mesh3D<Real> *mesh, bool use_genus = true);
	////////////////////////////////////////////////
	int compute_genus(const std::vector<MeshLib::HE_face<Real> *> &component);
	////////////////////////////////////////////////
	MeshLib::Mesh3D<Real> *mesh_subdiv(MeshLib::Mesh3D<Real> *block_mesh);
	////////////////////////////////////////////////

protected:
	int block_num, resolution, subdiv_level, seed;
	float perturb_level, scale_range;
	bool use_ccsubdiv;
	MeshLib::Mesh3D<Real> *quad_mesh = 0;
	const int neighbor[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
	const int two_step_neighbor[12][3] = {
		{1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {-1, 1, 0}, {1, 0, 1}, {1, 0, -1}, {-1, 0, -1}, {-1, 0, 1}, {0, 1, 1}, {0, 1, -1}, {0, -1, -1}, {0, -1, 1}};
	const int three_step_neighbor[8][3] = {
		{1, 1, 1}, {1, -1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, 1, -1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, -1}};

	int mesh_genus = 0;
};

typedef BlockGen<double> blockgen_3d;
