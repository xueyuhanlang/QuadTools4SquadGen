#include <iostream>

#include "cxxopts.hpp"
#include "MeshSubdivision.h"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "looputil.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("SplitMesh", "SplitMesh (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                          //
            ("i,input", "input filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())        //
            ("o,output", "output filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())      //
            ("n,num", "target max number of facets", cxxopts::value<int>()->default_value("50000")) //
            ("s,subdiv", "subdivision level", cxxopts::value<int>()->default_value("0"))            //
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << '\n';
            return 0;
        }

        if (!result.count("i"))
        {
            std::cout << "Please specify input filename!\n";
            return 1;
        }
        auto inputfile = result["i"].as<std::string>();
        if (!std::filesystem::exists(inputfile))
        {
            std::cout << "The input file does not exist!\n";
            return 1;
        }
        MeshLib::Mesh3D<double> *mesh = load_mesh<double>(inputfile);
        if (!mesh || mesh->get_num_of_faces() == 0 || (!mesh->is_tri() && !mesh->is_quad()))
        {
            std::cout << "The input mesh is not a valid triangle or quad mesh!\n";
            return 1;
        }

        int div = mesh->is_tri() ? 1 : 2;

        auto num_facets = std::max(mesh->get_num_of_faces(), ptrdiff_t(1));
        auto target_num = std::max(1, result["n"].as<int>());
        auto L = static_cast<int>(std::ceil(std::log(static_cast<double>(target_num) / (div * num_facets)) / std::log(4.0)));
        auto outputfile = result.count("o") ? result["o"].as<std::string>() : std::filesystem::current_path().string() + "/output.ply";
        auto subdiv_level = std::max(L, result["s"].as<int>());

        MeshLib::Mesh3D<double> *newmesh = mesh;
        for (int i = 0; i < subdiv_level; i++)
        {
            MeshLib::MeshSubdivision<double> subdiv_(newmesh);
            auto subdiv_mesh = div == 1 ? subdiv_.SplitTri() : subdiv_.SplitQuad();
            std::swap(subdiv_mesh, newmesh);
            if (subdiv_mesh != mesh)
                delete subdiv_mesh;
        }
        if (div == 1)
            save_mesh(newmesh, outputfile);
        else
        {
            std::vector<TinyVector<double, 3>> vertices;
            std::vector<std::vector<size_t>> face_indices, triface_indices;
            mesh_to_vertices_and_faces(newmesh, vertices, face_indices);
            triface_indices.resize(face_indices.size() * 2);
            for (size_t i = 0; i < face_indices.size(); i++)
            {
                if (split_quad(vertices[face_indices[i][0]], vertices[face_indices[i][1]], vertices[face_indices[i][2]], vertices[face_indices[i][3]]))
                {
                    triface_indices[2 * i] = {face_indices[i][0], face_indices[i][1], face_indices[i][2]};
                    triface_indices[2 * i + 1] = {face_indices[i][2], face_indices[i][3], face_indices[i][0]};
                }
                else
                {
                    triface_indices[2 * i] = {face_indices[i][1], face_indices[i][2], face_indices[i][3]};
                    triface_indices[2 * i + 1] = {face_indices[i][3], face_indices[i][0], face_indices[i][1]};
                }
            }
            save_mesh(vertices, triface_indices, outputfile);
        }
        if (newmesh != mesh)
            delete newmesh;
        delete mesh;
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
