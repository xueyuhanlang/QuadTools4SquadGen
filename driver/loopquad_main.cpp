#include <iostream>

#include "cxxopts.hpp"

#include "loopquad.h"
#include "MeshWriter.h"
#include "MeshLoader.h"
#include <filesystem>

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("LoopQuad", "LoopQuad (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                                                   //
            ("i,input", "input filename (*.obj;*.off;*.ply;*.glb;*.stl)", cxxopts::value<std::string>())                     //
            ("o,output", "output filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())                               //
            ("q,quality", "loop quality threshold", cxxopts::value<double>()->default_value("0.8"))                          //
            ("d,debug", "output nonquad mesh", cxxopts::value<bool>()->default_value("false"))                          // don't set it to false
            ("s,subdiv", "perform subdivision instead of merging triangles", cxxopts::value<bool>()->default_value("false")) // don't set it to true
            ("f,fast", "fast mode for handle nonmanifold ply", cxxopts::value<bool>()->default_value("true"))                // don't set it to false
            ("n,nonmanifold", "process nonmanifold ply", cxxopts::value<bool>()->default_value("true"))                      // don't set it to false
            ("m,minfacenum", "minimum face number for merging", cxxopts::value<int>()->default_value("16"))                  //
            ("h,help", "Print help");                                                                                        //

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
        auto nonmanifold = result["n"].as<bool>();

        auto inputfile = result["i"].as<std::string>();
        auto outputfile = result["o"].as<std::string>();

        auto mesh = load_mesh<double>(inputfile, nonmanifold);
        if (!mesh)
            return 1;

        auto loop_threshold = result["q"].as<double>();
        auto debug_tag = result["d"].as<bool>();
        if (debug_tag)
        {
            loop_threshold = 0;
        }
        auto subdiv_tag = result["s"].as<bool>();
        auto min_face_num = result["m"].as<int>();
        auto merged_mesh = LoopQuadProcessing<double>(mesh, loop_threshold, debug_tag, subdiv_tag, min_face_num);
        if (merged_mesh)
        {
            if (merged_mesh->get_num_of_faces() == 0)
            {
                delete merged_mesh;
                return 1;
            }
            save_mesh(merged_mesh, outputfile);
            delete merged_mesh;
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }
    return 0;
}

