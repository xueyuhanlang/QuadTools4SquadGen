#include "blockgen.h"

#include <unordered_map>
#include <ctime>
#include "MeshSubdivision.h"
#include <omp.h>
#include "MeshLoader.h"
#include "MeshWriter.h"

template <typename Real>
BlockGen<Real>::BlockGen(int _block_num, int _resolution, bool _use_ccsubdiv, int _subdiv_level,
                         float _perturb_level, float _scale_range, int _seed)
    : block_num(_block_num), resolution(_resolution), subdiv_level(_subdiv_level), perturb_level(_perturb_level), scale_range(_scale_range),
      use_ccsubdiv(_use_ccsubdiv), seed(_seed)
{
    perturb_level = std::max(0.0f, perturb_level);
    scale_range = std::max(1.0f, scale_range);
    block_num = std::max(1, block_num);
    resolution = std::max(1, resolution);
    subdiv_level = std::max(0, subdiv_level);

    auto block_mesh = create_block_mesh();
    if (block_mesh)
        quad_mesh = mesh_subdiv(block_mesh);
    else
        quad_mesh = 0;
}
template <typename Real>
BlockGen<Real>::~BlockGen()
{
    if (quad_mesh)
        delete quad_mesh;
}

template <typename Real>
void BlockGen<Real>::export_ply(const std::string &filename)
{
    SavePLYmesh_with_float_storage(quad_mesh, filename.c_str());
}
template <typename Real>
void BlockGen<Real>::export_obj(const std::string &filename)
{
    quad_mesh->write_obj(filename.c_str());
}
template <typename Real>
int BlockGen<Real>::get_genus()
{
    return mesh_genus;
}
////////////////////////////////////////////////
template <typename Real>
void BlockGen<Real>::insert_empty_block(int x, int y, int z,
                                        const std::vector<bool> &gridtag,
                                        const std::unordered_set<MySortedTuple<int, 3, false>> &empty_blocks_set,
                                        std::unordered_set<MySortedTuple<int, 3, false>> &added_empty_blocks_set)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
            {
                if (x + i >= 0 && x + i < resolution && y + j >= 0 && y + j < resolution && z + k >= 0 && z + k < resolution)
                {
                    if (!gridtag[(x + i) * resolution * resolution + (y + j) * resolution + z + k])
                    {
                        auto index = MySortedTuple<int, 3, false>(x + i, y + j, z + k);
                        if (empty_blocks_set.find(index) == empty_blocks_set.end())
                        {
                            added_empty_blocks_set.insert(index);
                        }
                    }
                }
            }
}
////////////////////////////////////////////////
template <typename Real>
void BlockGen<Real>::remove_block_region(int x, int y, int z,
                                         std::vector<bool> &gridtag,
                                         std::vector<MySortedTuple<int, 3, false>> &empty_blocks,
                                         std::unordered_set<MySortedTuple<int, 3, false>> &empty_blocks_set)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                if (range_check(x + i, y + j, z + k))
                {
                    auto tag = gridtag[(x + i) * resolution * resolution + (y + j) * resolution + z + k];
                    if (tag)
                    {
                        empty_blocks.push_back(MySortedTuple<int, 3, false>(x + i, y + j, z + k));
                        empty_blocks_set.insert(MySortedTuple<int, 3, false>(x + i, y + j, z + k));
                        gridtag[(x + i) * resolution * resolution + (y + j) * resolution + z + k] = false;
                    }
                }
}
/////////////////////////////////////////////////
template <typename Real>
bool BlockGen<Real>::range_check(int x, int y, int z)
{
    return x >= 0 && x < resolution && y >= 0 && y < resolution && z >= 0 && z < resolution;
}
template <typename Real>
bool BlockGen<Real>::one_ring_check(int x, int y, int z, int nx, int ny, int nz)
{
    return abs(nx - x) <= 1 && abs(ny - y) <= 1 && abs(nz - z) <= 1;
}
template <typename Real>
bool BlockGen<Real>::not_same_block(int x, int y, int z, int nx, int ny, int nz)
{
    return x != nx || y != ny || z != nz;
}
/////////////////////////////////////////////////
template <typename Real>
bool BlockGen<Real>::check_cells(bool use_block, // true for non empty block, false for empty block
                                 std::vector<bool> &gridtag,
                                 std::vector<MySortedTuple<int, 3, false>> &nonempty_blocks,
                                 std::vector<MySortedTuple<int, 3, false>> &empty_blocks,
                                 std::unordered_set<MySortedTuple<int, 3, false>> &empty_blocks_set)
{

    bool has_nonmanifold_issue = false;

    auto &cur_blocks = use_block ? nonempty_blocks : empty_blocks;
    auto bsize = cur_blocks.size();

    std::unordered_set<MySortedTuple<int, 3, false>> added_empty_blocks_set;

    bool detected = false;
    for (ptrdiff_t b = bsize - 1; b >= 0; b--)
    {
        int x = cur_blocks[b][0];
        int y = cur_blocks[b][1];
        int z = cur_blocks[b][2];

        if (use_block && gridtag[x * resolution * resolution + y * resolution + z] == false)
        {
            nonempty_blocks.erase(nonempty_blocks.begin() + b);
            continue;
        }
        std::unordered_set<MySortedTuple<int, 3, false>> one_step_neighbors, two_step_neighbors, three_step_neighbors;
        for (int nb = 0; nb < 6; nb++)
        {
            int nx = x + neighbor[nb][0];
            int ny = y + neighbor[nb][1];
            int nz = z + neighbor[nb][2];

            if (range_check(nx, ny, nz) &&
                gridtag[nx * resolution * resolution + ny * resolution + nz] == use_block)
            {
                one_step_neighbors.insert(MySortedTuple<int, 3, false>(nx, ny, nz));
            }
        }
        for (auto n : one_step_neighbors)
        {
            int nx = n[0];
            int ny = n[1];
            int nz = n[2];
            for (int nb = 0; nb < 6; nb++)
            {
                int nnx = nx + neighbor[nb][0];
                int nny = ny + neighbor[nb][1];
                int nnz = nz + neighbor[nb][2];
                auto nn = MySortedTuple<int, 3, false>(nnx, nny, nnz);
                if (not_same_block(nnx, nny, nnz, x, y, z) && range_check(nnx, nny, nnz) && one_ring_check(nnx, nny, nnz, x, y, z) &&
                    gridtag[nnx * resolution * resolution + nny * resolution + nnz] == use_block &&
                    one_step_neighbors.find(nn) == one_step_neighbors.end())
                {
                    two_step_neighbors.insert(nn);
                }
            }
        }

        for (auto n : two_step_neighbors)
        {
            int nx = n[0];
            int ny = n[1];
            int nz = n[2];
            for (int nb = 0; nb < 6; nb++)
            {
                int nnx = nx + neighbor[nb][0];
                int nny = ny + neighbor[nb][1];
                int nnz = nz + neighbor[nb][2];
                auto nn = MySortedTuple<int, 3, false>(nnx, nny, nnz);
                if (not_same_block(nnx, nny, nnz, x, y, z) && range_check(nnx, nny, nnz) && one_ring_check(nnx, nny, nnz, x, y, z) &&
                    gridtag[nnx * resolution * resolution + nny * resolution + nnz] == use_block &&
                    one_step_neighbors.find(nn) == one_step_neighbors.end() && two_step_neighbors.find(nn) == two_step_neighbors.end())
                {
                    three_step_neighbors.insert(nn);
                }
            }
        }
        bool detected = false;
        for (int nb = 0; nb < 12; nb++)
        {
            int nx = x + two_step_neighbor[nb][0];
            int ny = y + two_step_neighbor[nb][1];
            int nz = z + two_step_neighbor[nb][2];

            if (range_check(nx, ny, nz) &&
                gridtag[nx * resolution * resolution + ny * resolution + nz] == use_block &&
                two_step_neighbors.find(MySortedTuple<int, 3, false>(nx, ny, nz)) == two_step_neighbors.end())
            {

                if (use_block)
                {
                    empty_blocks.push_back(cur_blocks[b]);
                    empty_blocks_set.insert(cur_blocks[b]);
                    nonempty_blocks.erase(nonempty_blocks.begin() + b);
                    gridtag[x * resolution * resolution + y * resolution + z] = false;
                }
                else
                {
                    remove_block_region(x, y, z, gridtag, empty_blocks, empty_blocks_set);
                }
                has_nonmanifold_issue = true;
                detected = true;
                break;
            }
        }
        if (detected)
            continue;

        for (int nb = 0; nb < 8; nb++)
        {
            int nx = x + three_step_neighbor[nb][0];
            int ny = y + three_step_neighbor[nb][1];
            int nz = z + three_step_neighbor[nb][2];
            if (range_check(nx, ny, nz) &&
                gridtag[nx * resolution * resolution + ny * resolution + nz] == use_block &&
                three_step_neighbors.find(MySortedTuple<int, 3, false>(nx, ny, nz)) == three_step_neighbors.end())
            {
                if (use_block)
                {
                    empty_blocks.push_back(cur_blocks[b]);
                    empty_blocks_set.insert(cur_blocks[b]);
                    nonempty_blocks.erase(nonempty_blocks.begin() + b);
                    gridtag[x * resolution * resolution + y * resolution + z] = false;
                }
                else
                {
                    remove_block_region(x, y, z, gridtag, empty_blocks, empty_blocks_set);
                }
                has_nonmanifold_issue = true;
                detected = true;
                break;
            }
        }
    }

    empty_blocks_set.insert(added_empty_blocks_set.begin(), added_empty_blocks_set.end());
    empty_blocks.insert(empty_blocks.end(), added_empty_blocks_set.begin(), added_empty_blocks_set.end());

    return has_nonmanifold_issue;
}
/////////////////////////////////////////////////
template <typename Real>
bool BlockGen<Real>::check_block_removability(const MySortedTuple<int, 3, false> &block,
                                              std::vector<bool> &gridtag)
{
    return check_block_removability(block[0], block[1], block[2], gridtag);
}
/////////////////////////////////////////////////
template <typename Real>
bool BlockGen<Real>::check_block_removability(const int x, const int y, const int z,
                                              std::vector<bool> &gridtag)
{

    if (!range_check(x, y, z))
        return false;
    auto pos = x * resolution * resolution + y * resolution + z;
    if (!gridtag[pos])
        return false;

    // now set it to false to see whether  non-manifold is triggered
    gridtag[pos] = false;

    // check the neighbors
    std::unordered_set<MySortedTuple<int, 3, false>> one_step_neighbors, two_step_neighbors, three_step_neighbors;
    for (int nb = 0; nb < 6; nb++)
    {
        int nx = x + neighbor[nb][0];
        int ny = y + neighbor[nb][1];
        int nz = z + neighbor[nb][2];

        if (range_check(nx, ny, nz) &&
            gridtag[nx * resolution * resolution + ny * resolution + nz] == false)
        {
            one_step_neighbors.insert(MySortedTuple<int, 3, false>(nx, ny, nz));
        }
    }

    for (auto n : one_step_neighbors)
    {
        int nx = n[0];
        int ny = n[1];
        int nz = n[2];
        for (int nb = 0; nb < 6; nb++)
        {
            int nnx = nx + neighbor[nb][0];
            int nny = ny + neighbor[nb][1];
            int nnz = nz + neighbor[nb][2];
            auto nn = MySortedTuple<int, 3, false>(nnx, nny, nnz);
            if (not_same_block(nnx, nny, nnz, x, y, z) && range_check(nnx, nny, nnz) && one_ring_check(nnx, nny, nnz, x, y, z) &&
                gridtag[nnx * resolution * resolution + nny * resolution + nnz] == false &&
                one_step_neighbors.find(nn) == one_step_neighbors.end())
            {
                two_step_neighbors.insert(nn);
            }
        }
    }

    for (auto n : two_step_neighbors)
    {
        int nx = n[0];
        int ny = n[1];
        int nz = n[2];
        for (int nb = 0; nb < 6; nb++)
        {
            int nnx = nx + neighbor[nb][0];
            int nny = ny + neighbor[nb][1];
            int nnz = nz + neighbor[nb][2];
            auto nn = MySortedTuple<int, 3, false>(nnx, nny, nnz);
            if (not_same_block(nnx, nny, nnz, x, y, z) && range_check(nnx, nny, nnz) && one_ring_check(nnx, nny, nnz, x, y, z) &&
                gridtag[nnx * resolution * resolution + nny * resolution + nnz] == false &&
                one_step_neighbors.find(nn) == one_step_neighbors.end() && two_step_neighbors.find(nn) == two_step_neighbors.end())
            {
                three_step_neighbors.insert(nn);
            }
        }
    }

    for (int nb = 0; nb < 12; nb++)
    {
        int nx = x + two_step_neighbor[nb][0];
        int ny = y + two_step_neighbor[nb][1];
        int nz = z + two_step_neighbor[nb][2];

        if (range_check(nx, ny, nz) &&
            gridtag[nx * resolution * resolution + ny * resolution + nz] == false &&
            two_step_neighbors.find(MySortedTuple<int, 3, false>(nx, ny, nz)) == two_step_neighbors.end())
        {
            gridtag[pos] = true;
            return false;
        }
    }

    for (int nb = 0; nb < 8; nb++)
    {
        int nx = x + three_step_neighbor[nb][0];
        int ny = y + three_step_neighbor[nb][1];
        int nz = z + three_step_neighbor[nb][2];
        if (range_check(nx, ny, nz) &&
            gridtag[nx * resolution * resolution + ny * resolution + nz] == false &&
            three_step_neighbors.find(MySortedTuple<int, 3, false>(nx, ny, nz)) == three_step_neighbors.end())
        {
            gridtag[pos] = true;
            return false;
        }
    }

    gridtag[pos] = true; // restore the gridtag, let the external function to remove it
    return true;
}
/////////////////////////////////////////////////
template <typename Real>
bool BlockGen<Real>::next_cell(int x, int y, int z, int axis, int dir, int &nx, int &ny, int &nz, std::vector<bool> &gridtag)
{
    if (axis == 0)
    {
        nx = x + dir;
        ny = y;
        nz = z;
    }
    else if (axis == 1)
    {
        nx = x;
        ny = y + dir;
        nz = z;
    }
    else
    {
        nx = x;
        ny = y;
        nz = z + dir;
    }
    return check_block_removability(nx, ny, nz, gridtag);
}
/////////////////////////////////////////////////
template <typename Real>
bool BlockGen<Real>::is_interior_block(int x, int y, int z, const std::vector<bool> &gridtag)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                if (range_check(x + i, y + j, z + k))
                {
                    auto tag = gridtag[(x + i) * resolution * resolution + (y + j) * resolution + z + k];
                    if (!tag)
                        return false;
                }
                else
                    return false;
    return true;
}
/////////////////////////////////////////////////
template <typename Real>
void BlockGen<Real>::choose_axis_and_direction(int x, int y, int z, int &axis, int &direction, const std::vector<bool> &gridtag)
{
    for (int i = 0; i < 3; i++)
    {
        int nx = x + neighbor[3 * i][0];
        int ny = y + neighbor[3 * i][1];
        int nz = z + neighbor[3 * i][2];
        bool empty_tag_0 = !range_check(nx, ny, nz) || !gridtag[nx * resolution * resolution + ny * resolution + nz];
        nx = x + neighbor[3 * i + 1][0];
        ny = y + neighbor[3 * i + 1][1];
        nz = z + neighbor[3 * i + 1][2];
        bool empty_tag_1 = !range_check(nx, ny, nz) || !gridtag[nx * resolution * resolution + ny * resolution + nz];
        if (empty_tag_0 && !empty_tag_1)
        {
            axis = i;
            direction = 1;
            return;
        }
        else if (!empty_tag_0 && empty_tag_1)
        {
            axis = i;
            direction = -1;
            return;
        }
    }
    axis = 0, direction = 1; // this line should not be reached
}

/////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *BlockGen<Real>::create_block_mesh()
{
    std::vector<bool> gridtag(resolution * resolution * resolution, false);
    std::vector<MySortedTuple<int, 3, false>> blocks, empty_blocks;
    blocks.reserve(block_num);

    if (seed == -1)
        srand((unsigned int)time(NULL));
    else
        srand(seed);

    // create the first block

    auto origin = MySortedTuple<int, 3, false>(resolution / 2, resolution / 2, resolution / 2);
    gridtag[resolution / 2 * resolution * resolution + resolution / 2 * resolution + resolution / 2] = true;
    blocks.push_back(origin);
    // create blocks
    int trial_num = 0, max_trial_num = block_num * 100;
    while (blocks.size() < block_num && trial_num < max_trial_num)
    {
        // randomly select an entry from the grid_vec
        int idx = rand() % blocks.size();
        int x = blocks[idx][0];
        int y = blocks[idx][1];
        int z = blocks[idx][2];

        int rand_neighbor = rand() % 6;
        x = x + neighbor[rand_neighbor][0];
        y = y + neighbor[rand_neighbor][1];
        z = z + neighbor[rand_neighbor][2];
        auto pos = x * resolution * resolution + y * resolution + z;
        if (range_check(x, y, z) && !gridtag[pos])
        {
            gridtag[pos] = true;
            blocks.push_back(MySortedTuple<int, 3, false>(x, y, z));
        }
        trial_num++;
    }

    // resolve nonmanifold issue
    std::unordered_set<MySortedTuple<int, 3, false>> empty_blocks_set, new_empty_blocks_set;

    for (auto b : blocks)
    {
        insert_empty_block(b[0], b[1], b[2], gridtag, empty_blocks_set, new_empty_blocks_set);
    }
    empty_blocks.assign(new_empty_blocks_set.begin(), new_empty_blocks_set.end());
    std::swap(empty_blocks_set, new_empty_blocks_set);

    bool processed = true;
    while (processed)
    {
        processed = false;
        processed = check_cells(true, gridtag, blocks, empty_blocks, empty_blocks_set);
        if (!processed)
            processed = check_cells(false, gridtag, blocks, empty_blocks, empty_blocks_set);
    }

    // create random tunnels
    // int num_max_trials = blocks.size() / 2;
    int num_max_trials = rand() % blocks.size();
    size_t remove_counter = 0;

    int num_trials = 0;
    int num_tunnels = 0;
    while (num_trials < num_max_trials)
    {
        num_trials++;
        auto block_id = rand() % blocks.size();
        const auto &cell = blocks[block_id];
        int x = cell[0], y = cell[1], z = cell[2];
        if (
            is_interior_block(x, y, z, gridtag) ||
            !check_block_removability(cell, gridtag))
            continue;

        auto pos = x * resolution * resolution + y * resolution + z;
        gridtag[pos] = false;
        remove_counter++;

        int axis = rand() % 3, direction = 1;
        choose_axis_and_direction(x, y, z, axis, direction, gridtag);

        int cur_x = x, cur_y = y, cur_z = z;
        int nx, ny, nz;
        while (next_cell(cur_x, cur_y, cur_z, axis, direction, nx, ny, nz, gridtag))
        {
            cur_x = nx, cur_y = ny, cur_z = nz;
            auto pos = cur_x * resolution * resolution + cur_y * resolution + cur_z;
            gridtag[pos] = false;
            remove_counter++;
        }
        num_tunnels++;
    }
    // std::cout << "Create " << num_tunnels << " tunnels!" << std::endl;
    // std::cout << num_trials << " trials to create tunnels!" << std::endl;
    // std::cout << "Remove " << remove_counter << " blocks to create tunnels!" << std::endl;

    if (blocks.size() == remove_counter)
    {
        std::cerr << "No valid block is generated!" << std::endl;
        exit(0);
    }

    // create mesh
    MeshLib::Mesh3D<Real> *block_mesh = new MeshLib::Mesh3D<Real>;
    std::vector<ptrdiff_t> facevec;
    std::unordered_map<MySortedTuple<int, 3, false>, MeshLib::HE_vert<Real> *> mesh_vertices;
    int vertex_shift[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    int face_indices[6][4] = {{0, 3, 2, 1}, {1, 2, 6, 5}, {5, 6, 7, 4}, {4, 7, 3, 0}, {0, 1, 5, 4}, {3, 7, 6, 2}};
    int cell_shift[6][3] = {{0, 0, -1}, {1, 0, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 1, 0}};
    for (auto &block : blocks)
    {
        int x = block[0];
        int y = block[1];
        int z = block[2];

        if (!gridtag[x * resolution * resolution + y * resolution + z])
            continue;

        for (int f = 0; f < 6; f++)
        {
            auto opp_cell_index = MySortedTuple<int, 3, false>(x + cell_shift[f][0], y + cell_shift[f][1], z + cell_shift[f][2]);
            int ox = opp_cell_index[0];
            int oy = opp_cell_index[1];
            int oz = opp_cell_index[2];

            if (ox < 0 || ox >= resolution || oy < 0 || oy >= resolution || oz < 0 || oz >= resolution || !gridtag[ox * resolution * resolution + oy * resolution + oz])
            {
                std::vector<MeshLib::HE_vert<Real> *> vertex_list;
                for (int j = 0; j < 4; j++)
                {
                    auto index = MySortedTuple<int, 3, false>(x + vertex_shift[face_indices[f][j]][0], y + vertex_shift[face_indices[f][j]][1], z + vertex_shift[face_indices[f][j]][2]);
                    auto mesh_vertex = mesh_vertices.find(index);
                    if (mesh_vertex == mesh_vertices.end())
                    {
                        auto vertex = TinyVector<Real, 3>(index[0] - resolution / 2, index[1] - resolution / 2, index[2] - resolution / 2) / (resolution / 2);

                        mesh_vertices[index] = block_mesh->insert_vertex(vertex);
                        vertex_list.push_back(mesh_vertices[index]);
                    }
                    else
                    {
                        vertex_list.push_back(mesh_vertex->second);
                    }
                }
                for (auto v : vertex_list)
                    facevec.push_back(v->id);
                block_mesh->insert_face(vertex_list);
            }
        }
    }

    // block_mesh->update_mesh();
    // block_mesh->write_obj("block0.obj");
    auto mesh = pick_max_mesh_compoent(block_mesh);
    if (mesh != block_mesh)
        delete block_mesh;

    int irregular_count = 0;
#pragma omp parallel for reduction(+ : irregular_count)
    for (auto i = 0; i < mesh->get_num_of_vertices(); i++)
    {
        auto vertex = mesh->get_vertex(i);
        if (vertex->degree != 4)
            irregular_count++;
    }
    std::cout << "Irreguar vertices: " << irregular_count << std::endl;

    return mesh;
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *BlockGen<Real>::pick_max_mesh_compoent(MeshLib::Mesh3D<Real> *mesh, bool use_genus)
{
    int num_components = 0;
    std::vector<MeshLib::HE_face<Real> *> big_component;
    int cur_genus = -1;
    std::vector<int> genus_store;
    mesh->reset_faces_tag(false);
    for (int i = 0; i < mesh->get_num_of_faces(); i++)
    {
        auto face = mesh->get_face(i);
        if (face->tag)
            continue;

        std::queue<MeshLib::HE_face<Real> *> facequeue;
        facequeue.push(face);
        face->tag = true;
        std::vector<MeshLib::HE_face<Real> *> component;
        component.push_back(face);
        while (!facequeue.empty())
        {
            auto f = facequeue.front();
            facequeue.pop();

            auto he = f->edge;
            do
            {
                if (he->pair->face && !he->pair->face->tag)
                {
                    facequeue.push(he->pair->face);
                    he->pair->face->tag = true;
                    component.push_back(he->pair->face);
                }
                he = he->next;
            } while (he != f->edge);
        }
        if (use_genus)
        {
            int genus = compute_genus(component);
            if (genus > cur_genus || (genus == cur_genus && component.size() > big_component.size()))
            {
                std::swap(component, big_component);
                cur_genus = genus;
            }
            genus_store.push_back(genus);
        }
        else
        {
            if (component.size() > big_component.size())
                std::swap(component, big_component);
        }

        num_components++;
    }

    // for (auto g : genus_store)
    // {
    // 	std::cout << g << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "num_components: " << num_components << std::endl;
    // std::cout << "genus: " << cur_genus << std::endl;
    mesh_genus = cur_genus;
    if (num_components == 1)
        return mesh;
    else
        return create_mesh(big_component);
}
////////////////////////////////////////////////
template <typename Real>
int BlockGen<Real>::compute_genus(const std::vector<MeshLib::HE_face<Real> *> &component)
{
    std::unordered_set<MeshLib::HE_vert<Real> *> vert_set;
    for (auto f : component)
    {
        auto he = f->edge;
        do
        {
            vert_set.insert(he->vert);
            he = he->next;
        } while (he != f->edge);
    }
    size_t num_faces = component.size();
    size_t num_vertices = vert_set.size();
    size_t num_edges = num_faces * 2;
    int genus = (int)(2 + num_edges - num_vertices - num_faces) / 2;
    return genus;
}
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *BlockGen<Real>::mesh_subdiv(MeshLib::Mesh3D<Real> *block_mesh)
{
    if (scale_range != 1.0)
    {
        auto x_scale = (Real)rand() / RAND_MAX;
        auto y_scale = (Real)rand() / RAND_MAX;
        auto z_scale = (Real)rand() / RAND_MAX;
        x_scale = x_scale * scale_range + (1 - x_scale);
        y_scale = y_scale * scale_range + (1 - y_scale);
        z_scale = z_scale * scale_range + (1 - z_scale);

        for (ptrdiff_t i = 0; i < block_mesh->get_num_of_vertices(); i++)
        {
            auto vertex = block_mesh->get_vertex(i);
            vertex->pos[0] *= x_scale;
            vertex->pos[1] *= y_scale;
            vertex->pos[2] *= z_scale;
        }
    }
    for (int l = 0; l < subdiv_level; l++)
    {
        // apply jitter
        for (ptrdiff_t i = 0; i < block_mesh->get_num_of_vertices(); i++)
        {
            TinyVector<Real, 3> V((Real)rand() / RAND_MAX - 0.5, (Real)rand() / RAND_MAX - 0.5, (Real)rand() / RAND_MAX - 0.5);
            block_mesh->get_vertex(i)->pos += ((Real)perturb_level / resolution) * V;
        }
        MeshLib::MeshSubdivision<Real> subdiv(block_mesh);
        auto subdivmesh = use_ccsubdiv ? subdiv.Catmull_Clark() : subdiv.SplitQuad();
        delete block_mesh;
        block_mesh = subdivmesh;
    }

    return block_mesh;
}

//////////////////////////////////////////////////
template class BlockGen<double>;
