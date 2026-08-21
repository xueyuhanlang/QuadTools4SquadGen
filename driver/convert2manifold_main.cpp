#include <iostream>

#include "cxxopts.hpp"
#include "MeshWriter.h"
#include "MeshLoader.h"
#include <filesystem>
#include "NonmanifoldProcess.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("Convert2Manifold", "Convert2Manifold (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                               //
            ("i,input", "input filename (*.obj;*.off;*.ply;*.glb;*.stl)", cxxopts::value<std::string>()) //
            ("o,output", "output filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())           //
            ("h,help", "Print help");                                                                    //

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << '\n';
            return 0;
        }

        if (result.count("i") == 0)
        {
            std::cout << "No input file!\nPlease use -i option\n";
            return 1;
        }
        if (result.count("o") == 0)
        {
            std::cout << "No output file!\nPlease use -o option\n";
        }

        auto inputfile = result["i"].as<std::string>();
        auto outputfile = result["o"].as<std::string>();

        std::vector<TinyVector<double, 3>> vertices;
        std::vector<std::vector<size_t>> face_indices;
        mesh_load_interface(inputfile, vertices, face_indices, false);

        size_t original_face_count = face_indices.size();
        size_t original_vertex_count = vertices.size();
        nonmanifold_merge(vertices, face_indices);
        if (!face_indices.empty())
        {
            if (face_indices.size() != original_face_count || vertices.size() != original_vertex_count || inputfile != outputfile)
                save_mesh(vertices, face_indices, outputfile);
        }
        else
        {
            if (inputfile == outputfile)
            {
                std::filesystem::remove(inputfile);
            }
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
