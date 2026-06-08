#include <iostream>
#include <filesystem>
#include "cxxopts.hpp"
#include "MeshLoader.h"
#include "myutils.h"
#include "qcdf2quadmesh.h"
#include "cdfdcdf2quadmesh.h"

int main(int argc, char **argv)
{
    MeshLib::Mesh3D<double> *mesh = 0;
    try
    {
        cxxopts::Options options("QuadExtraction", "QuadExtraction (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            // .allow_unrecognised_options()
            .add_options()                                                                    //
            ("i,input", "input filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())  //
            ("f,feature", "feature input (*.npz)", cxxopts::value<std::string>())             //
            ("o,output", "extracted mesh (*.obj;*.off;*.ply)", cxxopts::value<std::string>()) //
            ("t,threshold", "threshold for color binarization", cxxopts::value<float>()->default_value("0.1")) // for noise data, use a smaller value to get better result
            ("s,subdiv", "subdiv num", cxxopts::value<unsigned int>()->default_value("0"))                     //
            ("r,smooth", "smooth num", cxxopts::value<unsigned int>()->default_value("0"))                     //
            ("a,sharpangle", "sharp feature angle in degree", cxxopts::value<float>()->default_value("150"))   //
            ("d,debug", "debug mode", cxxopts::value<bool>()->default_value("false"))                          //
            ("m,method", "cdf, dcdf, cdfdcdf, all", cxxopts::value<std::string>()->default_value("cdfdcdf"))   //
            ("ringsize", "ring size for color enhancement", cxxopts::value<int>()->default_value("10"))                                //
            ("c,collasperatio", "edge collapse ratio", cxxopts::value<float>()->default_value("0.05"))                                 //
            ("collasenormalthreshold", "edge collapse normal angle threshold in degree", cxxopts::value<float>()->default_value("20")) //
            ("n,normalize", "normalize the input mesh", cxxopts::value<bool>()->default_value("false"))                                 //
            ("improve", "improve quad numbers", cxxopts::value<bool>()->default_value("false"))                                        //
            ("v,verbose", "verbose output", cxxopts::value<bool>()->default_value("true"))  //
            ("div", "color pattern division", cxxopts::value<int>()->default_value("0"))    //
            ("u,useqcdf2", "use QCDF code", cxxopts::value<bool>()->default_value("false")) //
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        if (result.count("i"))
        {
            auto inputfile = result["i"].as<std::string>();

            if (!std::filesystem::exists(inputfile))
            {
                std::cout << "The input file does not exist!" << std::endl;
                exit(0);
            }

            mesh = load_mesh<double>(inputfile);
            if (!mesh)
            {
                std::cout << "The input file " << inputfile << "is loaded incorrectly(non - manifold)!" << std::endl;
                exit(0);
            }

            std::string featurefile;
            if (result.count("f"))
            {
                featurefile = result["f"].as<std::string>();
                if (GetFileExtension(featurefile) != "npz")
                {
                    std::cout << "The input file should be a npz file!" << std::endl;
                    throw std::invalid_argument("The input file should be a npz file!");
                }
                if (!std::filesystem::exists(featurefile))
                {
                    std::cout << "The feature file does not exist!" << std::endl;
                    throw std::invalid_argument("The feature file does not exist!");
                }
            }
            else
            {
                std::cout << "No feature file!" << std::endl;
                throw std::invalid_argument("No feature file!");
            }
            std::string outputfilename = result.count("o") ? result["o"].as<std::string>() : featurefile.substr(0, featurefile.find_last_of(".")) + "_quad.ply";
            std::string outputdir = GetFileDirectory(outputfilename);
            if (!outputdir.empty() && !std::filesystem::exists(outputdir))
            {
                std::filesystem::create_directories(outputdir);
            }

            try
            {
                if (!result["u"].as<bool>())
                {
                    CDFDCDF2QuadMesh<double> quadextractor(mesh, featurefile, result["div"].as<int>() > 0);
                    quadextractor.set_verbose(result["v"].as<bool>());
                    quadextractor.set_debug_mode(result["d"].as<bool>());
                    quadextractor.set_debug_dir(outputdir);
                    quadextractor.set_ring_size(result["ringsize"].as<int>());
                    quadextractor.set_normalize(result["n"].as<bool>());
                    quadextractor.set_confusion_band(result["t"].as<float>());
                    quadextractor.set_sharp_angle(result["a"].as<float>());
                    quadextractor.set_improve_mode(result["improve"].as<bool>());
                    quadextractor.set_subdiv_num(result["s"].as<unsigned int>());
                    quadextractor.set_smooth_num(result["r"].as<unsigned int>());
                    quadextractor.set_edge_collapse_normal_threshold(result["collasenormalthreshold"].as<float>());
                    quadextractor.set_edge_collapse_ratio(result["collasperatio"].as<float>());
                    quadextractor.export_mesh(outputfilename);
                }
                else
                {
                    auto method = result["m"].as<std::string>();
                    QCDF2QuadMesh<double> qcdf2quadextractor(mesh, featurefile, result["div"].as<int>() > 0);
                    qcdf2quadextractor.set_verbose(result["v"].as<bool>());
                    qcdf2quadextractor.set_debug_mode(result["d"].as<bool>());
                    qcdf2quadextractor.set_debug_dir(outputdir);
                    qcdf2quadextractor.set_ring_size(result["ringsize"].as<int>());
                    qcdf2quadextractor.set_normalize(result["n"].as<bool>());
                    qcdf2quadextractor.set_confusion_band(result["t"].as<float>());
                    qcdf2quadextractor.set_sharp_angle(result["a"].as<float>());
                    qcdf2quadextractor.set_improve_mode(result["improve"].as<bool>());
                    qcdf2quadextractor.set_subdiv_num(result["s"].as<unsigned int>());
                    qcdf2quadextractor.set_smooth_num(result["r"].as<unsigned int>());
                    qcdf2quadextractor.set_edge_collapse_normal_threshold(result["collasenormalthreshold"].as<float>());
                    qcdf2quadextractor.set_edge_collapse_ratio(result["collasperatio"].as<float>());

                    FaceClusterType method_type;
                    if (method == "cdf")
                        method_type = FaceClusterType::FC_CDF;
                    else if (method == "dcdf")
                        method_type = FaceClusterType::FC_DCDF;
                    else if (method == "cdfdcdf")
                        method_type = FaceClusterType::FC_CDF_DCDF;
                    else if (method == "all")
                        ;
                    else
                    {
                        std::cerr << "Unknown method type!" << std::endl;
                        throw std::invalid_argument("Unknown method type!");
                    }
                    if (method == "all")
                    {
                        std::string cdf_outputfilename = outputfilename.substr(0, outputfilename.find_last_of(".")) + "_cdf.ply";
                        qcdf2quadextractor.export_mesh(FaceClusterType::FC_CDF, cdf_outputfilename);
                        std::string dcdf_outputfilename = outputfilename.substr(0, outputfilename.find_last_of(".")) + "_dcdf.ply";
                        qcdf2quadextractor.export_mesh(FaceClusterType::FC_DCDF, dcdf_outputfilename);
                        std::string cdfdcdf_outputfilename = outputfilename.substr(0, outputfilename.find_last_of(".")) + "_cdfdcdf.ply";
                        qcdf2quadextractor.export_mesh(FaceClusterType::FC_CDF_DCDF, cdfdcdf_outputfilename);
                    }
                    else
                        qcdf2quadextractor.export_mesh(method_type, outputfilename);
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << std::endl;
    }

    if (mesh)
        delete mesh;

    return 0;
}
