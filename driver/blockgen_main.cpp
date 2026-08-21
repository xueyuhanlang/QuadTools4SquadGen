#include <iostream>

#include "cxxopts.hpp"
#include "blockgen.h"
#include "myutils.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("BlockGen2", "BlockGen2 (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                                                 //
            ("b,blocknum", "resolution (default: 8)", cxxopts::value<int>()->default_value("8"))                           //
            ("r,resolution", "resolution (default: 8)", cxxopts::value<int>()->default_value("8"))                         //
            ("o,output", "output filename (*.obj;*.ply)", cxxopts::value<std::string>())                                   //
            ("l,subdivlevel", "subdiv level (default: 2)", cxxopts::value<int>()->default_value("2"))                      //
            ("c,ccsubdiv", "use Catmull-Clark subdivision (default: true)", cxxopts::value<bool>()->default_value("true")) //
            ("p,perturb_level", "perturb_level (default: 0.2)", cxxopts::value<float>()->default_value("0.2"))             //
            ("s,scale_range", "random scaling (default: 1.0)", cxxopts::value<float>()->default_value("1.0"))              //
            ("seed", "random seed (default: 0)", cxxopts::value<int>()->default_value("0"))                                //
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << '\n';
            return 0;
        }

        if (result.count("o"))
        {
            auto filename      = result["o"].as<std::string>();
            auto block_num     = std::max(1, result["b"].as<int>());
            auto resolution    = std::max(1, result["r"].as<int>());
            auto subdiv_level  = std::max(1, result["l"].as<int>());
            auto scale_range   = std::fabs(result["s"].as<float>());
            auto perturb_level = result["p"].as<float>();
            auto seed          = result["seed"].as<int>();
            auto use_cc_subdiv = result["c"].as<bool>();

            blockgen_3d blockgen(block_num, resolution, use_cc_subdiv, subdiv_level, perturb_level, scale_range, seed);

            const int mesh_genus = blockgen.get_genus();
            std::cout << "Mesh genus: " << mesh_genus << '\n';

            const auto ext = GetFileExtension(filename);
            if (ext == "ply")
                blockgen.export_ply(filename);
            else
                blockgen.export_obj(filename);
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
