#include <iostream>
#include <filesystem>
#include "cxxopts.hpp"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "myutils.h"
#include <unordered_set>

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("FormatConvert", "FormatConvert (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                                                                                      //
            ("i,input", "input filename (*.ply,*.off,*.obj,*.glb,*.gltf,*.stl,*.wrl,*.fbx)", cxxopts::value<std::string>())                     //
            ("o,output", "output filename (*.ply,*.off,*.obj,*.glb,*.gltf,*.vtk,*.vtp,*.stl,*.usd,*.wrl,*.x3d)", cxxopts::value<std::string>()) //
            ("m,merge", "merge adjacent triangles into quads", cxxopts::value<bool>()->default_value("false"))                                  // handle quad conversion
            ("h,help", "Print help");

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        if (result.count("i") && result.count("o"))
        {
            auto inputfile = result["i"].as<std::string>();
            auto inputext = GetFileExtension(inputfile);

            if (!std::filesystem::exists(inputfile))
            {
                std::cout << "The input file does not exist!" << std::endl;
                exit(0);
            }

            const char *supported_input_formats = "ply, off, obj, stl, glb, gltf, vtk, vtp, wrl, fbx";
            const char *supported_output_formats = "ply, off, obj, stl, glb, gltf, vtk, vtp, usd, wrl, x3d";
            if (inputext.empty() || std::string::npos == std::string(supported_input_formats).find(inputext))
            {
                std::cout << "The input file format is not supported! Supported formats are: " << supported_input_formats << std::endl;
                exit(0);
            }

            auto outputfile = result["o"].as<std::string>();
            auto outputext = GetFileExtension(outputfile);
            if (outputext.empty() || std::string::npos == std::string(supported_output_formats).find(outputext))
            {
                std::cout << "The output file format is not supported! Supported formats are: " << supported_output_formats << std::endl;
                exit(0);
            }

            auto outputdir = GetFileDirectory(outputfile);
            if (!std::filesystem::exists(outputdir))
                std::filesystem::create_directories(outputdir);

            std::vector<TinyVector<double, 3>> vertices;
            std::vector<std::vector<size_t>> face_indices;
            mesh_load_interface(inputfile, vertices, face_indices, false);

            //rotate
            // for (auto &v : vertices)
            // {
            //     const auto temp = v;
            //     v[1] = temp[2], v[2] = -temp[1];
            // }

            if (result["m"].as<bool>())
            {
                // try to merge triangles into quads
                std::vector<std::vector<size_t>> merged_face_indices;
                merged_face_indices.reserve(face_indices.size() / 2);
                for (size_t i = 0; i < face_indices.size(); i += 2)
                {
                    std::unordered_set<size_t> vertex_set(face_indices[i + 1].begin(), face_indices[i + 1].end());
                    std::vector<size_t> merged_face;
                    for (int j = 0; j < 3; j++)
                    {
                        if (vertex_set.find(face_indices[i][j]) == vertex_set.end())
                        {
                            merged_face.push_back(face_indices[i][j]);
                            merged_face.push_back(face_indices[i][(j + 1) % 3]);
                            for (int k = 0; k < 3; k++)
                            {
                                if (face_indices[i + 1][k] != face_indices[i][(j + 1) % 3] && face_indices[i + 1][k] != face_indices[i][(j + 2) % 3])
                                {
                                    merged_face.push_back(face_indices[i + 1][k]);
                                    break;
                                }
                            }
                            merged_face.push_back(face_indices[i][(j + 2) % 3]);
                            break;
                        }
                    }
                    merged_face_indices.push_back(merged_face);
                }
                face_indices = merged_face_indices;
            }

            save_mesh(vertices, face_indices, outputfile);
        }
        else
        {
            std::cout << "No input file!" << std::endl;
            exit(0);
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << std::endl;
        exit(0);
    }
    return 0;
}
