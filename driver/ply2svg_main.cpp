#include <iostream>
#include <filesystem>
#include "cxxopts.hpp"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "myutils.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("PlY2SVG", "Ply2SVG (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                       //
            ("i,input", "input filename (*.ply)", cxxopts::value<std::string>()) //
            ("o,output", "svg output (*.svg)", cxxopts::value<std::string>())    //
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << '\n';
            return 0;
        }

        if (result.count("i"))
        {
            auto inputfile = result["i"].as<std::string>();

            if (!std::filesystem::exists(inputfile))
            {
                std::cout << "The input file does not exist!\n";
                return 1;
            }

            std::string ext = GetFileExtension(inputfile);
            MeshLib::Mesh3D<double> *mesh = nullptr;
            bool load_success = true;
            std::vector<std::array<unsigned char, 3>> ply_face_color;
            if (ext == "ply")
            {
                mesh = LoadPLYmesh<double>(inputfile.c_str(), ply_face_color);
                load_success = (mesh != nullptr);
            }
            else
            {
                std::cout << "Unsupported file format!\n";
                return 1;
            }

            if (!load_success)
            {
                std::cout << "The input file is loaded incorrectly (non-manifold)!\n";
                if (mesh)
                    delete mesh;
                return 1;
            }

            std::string outputfilename = result.count("o")
                ? result["o"].as<std::string>()
                : std::filesystem::path(inputfile).replace_extension(".svg").string();
            if (result.count("o"))
            {
                auto outputdir = GetFileDirectory(outputfilename);
                if (!std::filesystem::exists(outputdir))
                {
                    std::filesystem::create_directories(outputdir);
                }
            }

            save_as_svg(mesh, ply_face_color, outputfilename.c_str());

            if (mesh)
                delete mesh;
        }
        else
        {
            std::cout << "No input file!\n";
            return 1;
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
