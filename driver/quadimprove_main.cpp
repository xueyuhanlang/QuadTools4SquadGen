#include <iostream>
#include <filesystem>
#include "cxxopts.hpp"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "myutils.h"
#include "quadimprove.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("QuadImprove", "QuadImprove (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                     //
            ("i,input", "input filename (*.ply,*.off,*.obj)", cxxopts::value<std::string>())   //
            ("o,output", "output filename (*.ply,*.off,*.obj)", cxxopts::value<std::string>()) //
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << '\n';
            return 0;
        }

        if (result.count("i") && result.count("o"))
        {
            auto inputfile = result["i"].as<std::string>();
            auto inputext = GetFileExtension(inputfile);

            if (!std::filesystem::exists(inputfile))
            {
                std::cout << "The input file does not exist!\n";
                return 1;
            }

            const char *supported_input_formats = "ply, off, obj";
            const char *supported_output_formats = "ply, off, obj";
            if (inputext.empty() || std::string::npos == std::string(supported_input_formats).find(inputext))
            {
                std::cout << "The input file format is not supported! Supported formats are: " << supported_input_formats << '\n';
                return 1;
            }

            auto outputfile = result["o"].as<std::string>();
            auto outputext = GetFileExtension(outputfile);
            if (outputext.empty() || std::string::npos == std::string(supported_output_formats).find(outputext))
            {
                std::cout << "The output file format is not supported! Supported formats are: " << supported_output_formats << '\n';
                return 1;
            }

            auto outputdir = GetFileDirectory(outputfile);
            if (!std::filesystem::exists(outputdir))
                std::filesystem::create_directories(outputdir);

            std::vector<TinyVector<double, 3>> vertices;
            std::vector<std::vector<size_t>> face_indices;
            mesh_load_interface(inputfile, vertices, face_indices, false);
            save_mesh(vertices, face_indices, outputfile);
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
