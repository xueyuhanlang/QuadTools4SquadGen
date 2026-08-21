#include "cxxopts.hpp"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "patchsample.h"
#include "myutils.h"
#include "MeshSubdivision.h"
#include "MeshSmooth.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("CDFGen", "CDFGen (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                                                   //
            ("i,input", "input filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())                                 //
            ("d,dir", "point (npz) output directory", cxxopts::value<std::string>())                                         //
            ("f,fname", "fname", cxxopts::value<std::string>())                                                              //
            ("a,sharpangle", "sharp angle in degree <= 180 (180 means flat)", cxxopts::value<float>()->default_value("130")) //
            ("s,seed", "random seed", cxxopts::value<unsigned int>()->default_value("0"))                                    //
            ("n,npts", "number of points", cxxopts::value<int>()->default_value("50000"))                                    //
            ("fpsnum_min", "number of fps points", cxxopts::value<int>()->default_value("512"))                              //
            ("fpsnum_max", "number of fps points", cxxopts::value<int>()->default_value("4096"))                             //
            ("fpscopies", "number of fps copies", cxxopts::value<int>()->default_value("5"))                                 //
            ("q", "save quadextractioninfo", cxxopts::value<bool>()->default_value("false"))                                 // for debugging
            ("normalize", "normalize the model", cxxopts::value<bool>()->default_value("true"))                              //
            ("subdivnum", "subdivision num for npz output", cxxopts::value<int>()->default_value("0"))                       //
            ("ccsubdiv", "use Catmul-Clark subivision or QuadSplit", cxxopts::value<bool>()->default_value("false"))         //
            ("maxsubdivfacenum", "max subdivface limits for patchgen", cxxopts::value<int>()->default_value("50000"))        //
            ("usemeshascomplex", "use the input mesh as complex", cxxopts::value<bool>()->default_value("false"))            //
            ("r,resolution", "resolution", cxxopts::value<int>()->default_value("-1"))                                       //
            ("debug", "debug mode", cxxopts::value<bool>()->default_value("false"))                                          // for debugging
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
                std::cout << "Input file " << inputfile << " does not exist!\n";
                return 1;
            }

            MeshLib::Mesh3D<double> *mesh = load_mesh<double>(inputfile.c_str());
            if (!mesh)
            {
                std::cout << "Failed to load mesh from " << inputfile << '\n';
                return 1;
            }

            auto output_npz = result.count("d");
            if (!output_npz)
            {
                std::cout << "No output directory specified. Use -d option.\n";
                return 1;
            }
            auto npz_output_folder = output_npz ? result["d"].as<std::string>() : std::filesystem::current_path().string();

            if (output_npz && !std::filesystem::exists(npz_output_folder))
            {
                std::filesystem::create_directories(npz_output_folder);
            }
            auto seed = result["s"].as<unsigned int>();
            auto sharpangle = result["a"].as<float>();
            auto numpts = std::max(1, result["n"].as<int>());
            auto fpsnum_min = std::max(1, result["fpsnum_min"].as<int>());
            auto fpsnum_max = std::max(1, result["fpsnum_max"].as<int>());
            auto fpscopies = std::max(1, result["fpscopies"].as<int>());
            auto cc_subdiv = result["ccsubdiv"].as<bool>();
            auto quadextractioninfo = result["q"].as<bool>();
            auto model_normalize = result["normalize"].as<bool>();
            auto debug_mode = result["debug"].as<bool>();
            auto num_subdiv = std::max(0, result["subdivnum"].as<int>());
            auto maxsubdivfacenum = std::max(0, result["maxsubdivfacenum"].as<int>());
            auto usemeshascomplex = result["usemeshascomplex"].as<bool>();
            auto resolution = result["r"].as<int>();

            std::vector<MeshLib::Mesh3D<double> *> submeshes;
            mesh_decomposition<double>(mesh, submeshes);
            std::string filename_only = GetFileNameWithoutExtension(GetFileNameWithoutDirInfo(inputfile));
            if (result.count("f"))
            {
                filename_only = result["f"].as<std::string>();
            }
            int mesh_count = 0;
            for (auto &submesh : submeshes)
            {
                if (submesh->is_quad())
                {
                    MeshLib::Mesh3D<double> *subdivmesh = submesh;
                    if (num_subdiv > 0)
                    {
                        if (submesh->get_num_of_faces() > maxsubdivfacenum || subdivmesh->get_num_of_faces() * 4 > maxsubdivfacenum)
                            continue;
                        for (int iter = 0; iter < num_subdiv; iter++)
                        {
                            if (subdivmesh->get_num_of_faces() * 4 > maxsubdivfacenum)
                                break;
                            MeshLib::MeshSubdivision<double> subdiv_(subdivmesh);
                            auto new_quad_mesh = cc_subdiv ? subdiv_.Catmull_Clark() : subdiv_.SplitQuad();
                            std::swap(subdivmesh, new_quad_mesh);
                            if (new_quad_mesh != submesh)
                                delete new_quad_mesh;
                        }
                    }
                    try
                    {
                        PatchSample<double> patchsample(subdivmesh, numpts, fpsnum_min, fpsnum_max, fpscopies,
                                                        model_normalize, seed, usemeshascomplex, sharpangle, resolution, debug_mode);
                        std::string outputnpzfilename = npz_output_folder + "/" + filename_only + "_" + std::to_string(mesh_count) + ".npz";
                        patchsample.save_samples_to_npz(outputnpzfilename.c_str(), quadextractioninfo);
                    }
                    catch (const std::exception &e)
                    {
                        std::cout << "Error in processing " << inputfile << "; submesh_id: " << mesh_count << '\n';
                        std::cerr << e.what() << '\n';
                    }
                    if (subdivmesh != submesh)
                        delete subdivmesh;
                }
                mesh_count++;
                delete submesh;
            }

            if (mesh)
                delete mesh;
        }
        else
        {
            std::cout << "No input file! Use -i option.\n";
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
