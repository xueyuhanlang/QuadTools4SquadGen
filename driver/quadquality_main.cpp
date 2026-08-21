#include <iostream>
#include <iomanip>
#include <filesystem>
#include "cxxopts.hpp"
#include "quadquality.h"
#include "MeshLoader.h"
#include "MeshWriter.h"
#include "MeshSubdivision.h"
#include "termcolor.hpp"
#include "PointUtil.h"
#include <climits>

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("QuadQuality", "QuadQuality (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")                                                                                             //
            .show_positional_help()                                                                                                         //
            .allow_unrecognised_options()                                                                                                   //
            .add_options()                                                                                                                  //
            ("i,input", "input filename (*.obj;*.off;*.ply)", cxxopts::value<std::string>())                                                //
            ("o,output", "[dump submesh] output dir", cxxopts::value<std::string>())                                                        // (optional)
            ("s,submeshid", "submesh id", cxxopts::value<int>()->default_value("-1"))                                                       // output the specified submesh, only useful when option -o is specified
            ("n,normalize", "normalize the submesh", cxxopts::value<bool>()->default_value("true"))                                         // only useful when option -o is specified
            ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))                                                 //
            ("j,json", "output json folder", cxxopts::value<std::string>())                                                                 //
            ("d,dump", "[dump layout complex] dir", cxxopts::value<std::string>())                                                          // dump the compoent
            ("m,dumpmax", "dump layout complex for the component with the largest face num", cxxopts::value<bool>()->default_value("true")) //
            ("h,help", "Print help");                                                                                                       //

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << '\n';
            return 0;
        }

        bool decompose_and_dump = result.count("o") > 0;
        auto output_dir = decompose_and_dump ? result["o"].as<std::string>() : "";
        bool normalize_submesh = result["n"].as<bool>();
        int specified_submesh_id = result["s"].as<int>();
        bool dump_json = result.count("j") > 0;
        auto json_folder = dump_json ? result["j"].as<std::string>() : "";
        bool dump_complex = result.count("d") > 0;
        auto output_complex_dir = dump_complex ? result["d"].as<std::string>() : "";
        auto dump_largest_only = result["m"].as<bool>();
        if (result.count("i"))
        {
            auto inputfile = result["i"].as<std::string>();
            if (!std::filesystem::exists(inputfile))
            {
                return 0;
            }

            MeshLib::Mesh3D<double> *mesh = load_mesh<double>(inputfile);
            TinyVector<double, 3> normalization_center, inverse_rotation[3];
            double normalization_scale = 1;
            if (mesh)
            {
                auto verbose = result["v"].as<bool>();
                std::vector<MeshLib::Mesh3D<double> *> submeshes;
                mesh_decomposition(mesh, submeshes);
                const auto inputfile_stem = std::filesystem::path(inputfile).stem().string();
                if (decompose_and_dump)
                {
                    std::filesystem::create_directories(output_dir);
                    if (specified_submesh_id >= 0 && specified_submesh_id < static_cast<int>(submeshes.size()))
                    {
                        std::string submesh_filename = output_dir + "/" + inputfile_stem + "_submesh_" + std::to_string(specified_submesh_id) + ".ply";
                        if (normalize_submesh)
                            scale_and_PCA(submeshes[specified_submesh_id], inverse_rotation, normalization_center, normalization_scale);

                        save_mesh<double>(submeshes[specified_submesh_id], submesh_filename.c_str());
                    }
                    else
                    {
                        size_t submesh_id = 0;
                        for (auto submesh : submeshes)
                        {
                            if (normalize_submesh)
                                scale_and_PCA(submesh, inverse_rotation, normalization_center, normalization_scale);
                            std::string submesh_filename = output_dir + "/" + inputfile_stem + "_submesh_" + std::to_string(submesh_id++) + ".ply";
                            save_mesh<double>(submesh, submesh_filename.c_str());
                        }
                    }

                    return 0; // skip the quality computation
                }

                if (dump_json)
                    std::filesystem::create_directories(json_folder);
                if (dump_complex)
                    std::filesystem::create_directories(output_complex_dir);
                std::string json_filename = json_folder + "/" + inputfile_stem + "_quadquality.json";
                std::ofstream json_off(json_filename);
                json_off << std::fixed << std::setprecision(6);
                json_off << "[\n";

                ptrdiff_t largest_face_num = 0;
                int largest_submesh_id = -1;
                if (dump_largest_only)
                {
                    for (size_t i = 0; i < submeshes.size(); i++)
                    {
                        auto submesh = submeshes[i];
                        if (submesh->get_num_of_faces() > largest_face_num)
                        {
                            largest_face_num = submesh->get_num_of_faces();
                            largest_submesh_id = static_cast<int>(i);
                        }
                    }
                }

                int submesh_count = 0;
                for (auto submesh : submeshes)
                {
                    auto pointer = submesh;
                    if (!pointer->is_quad())
                    {
                        MeshLib::MeshSubdivision<double> subdiv_(submesh);
                        pointer = subdiv_.SplitQuad();
                    }
                    QuadQuality<double> quadquality(pointer, verbose);

                    if (dump_complex)
                    {
                        if (!dump_largest_only)
                        {
                            std::string complex_edge_ply_filename = output_complex_dir + "/" + inputfile_stem + "_submesh_" + std::to_string(submesh_count) + "_complex_edges.ply";
                            quadquality.export_base_complex_edges_as_ply(complex_edge_ply_filename.c_str());
                            std::string complex_face_ply_filename = output_complex_dir + "/" + inputfile_stem + "_submesh_" + std::to_string(submesh_count) + "_complex_faces.ply";
                            quadquality.export_base_complex_faces_as_ply(complex_face_ply_filename.c_str());
                        }
                        else if (submesh_count == largest_submesh_id)
                        {
                            std::string complex_edge_ply_filename = output_complex_dir + "/" + inputfile_stem + "_complex_edges.ply";
                            quadquality.export_base_complex_edges_as_ply(complex_edge_ply_filename.c_str());
                            std::string complex_face_ply_filename = output_complex_dir + "/" + inputfile_stem + "_complex_faces.ply";
                            quadquality.export_base_complex_faces_as_ply(complex_face_ply_filename.c_str());
                        }
                    }

                    auto fratio = quadquality.get_simple_faceloop_ratio(), eratio = quadquality.get_simple_edgeloop_ratio();
                    // if (std::isnan(fratio) || std::isnan(eratio))
                    // {
                    //     if (pointer != submesh)
                    //         delete pointer;
                    //     delete submesh;
                    //     continue;
                    // }

                    auto fratio_new = quadquality.get_simple_faceloop_ratio_new(), eratio_new = quadquality.get_simple_edgeloop_ratio_new();
                    auto fsprial_ratio = quadquality.get_faceloop_spriality_ratio(), esprial_ratio = quadquality.get_edgeloop_spriality_ratio();
                    double max_area_ratio, min_area_ratio, mean_area_ratio, min_edge_length;
                    quadquality.get_complex_distribution(max_area_ratio, min_area_ratio, mean_area_ratio, min_edge_length);
                    // auto size = std::max(pointer->xmax - pointer->xmin,
                    //                      std::max(pointer->ymax - pointer->ymin,
                    //                               pointer->zmax - pointer->zmin));
                    double size = pointer->xmax - pointer->xmin;
                    if (size < pointer->ymax - pointer->ymin)
                        size = pointer->ymax - pointer->ymin;
                    if (size < pointer->zmax - pointer->zmin)
                        size = pointer->zmax - pointer->zmin;
                    if (verbose)
                    {
                        std::cout << "area_ratio: max " << max_area_ratio << ", min " << min_area_ratio << ", mean " << mean_area_ratio
                                  << "; min_edge_length: " << min_edge_length / size << '\n';
                        static bool header_printed = false;
                        if (!header_printed)
                        {
                            std::cout << termcolor::color<211, 54, 130>;
                            std::cout << std::left
                                      << std::setw(14) << "FratioN"
                                      << std::setw(14) << "EratioN"
                                      << std::setw(14) << "Fspiral"
                                      << std::setw(14) << "Espiral"
                                      << std::setw(14) << "Fratio"
                                      << std::setw(14) << "Eratio"
                                      << std::setw(12) << "#Complex"
                                      << std::setw(12) << "#IrregVtx"
                                      << std::setw(12) << "#Vertices"
                                      << std::setw(12) << "#Edges"
                                      << std::setw(12) << "#Faces"
                                      << std::setw(14) << "#Boundaries"
                                      << std::setw(8) << "#Genus"
                                      << '\n';
                            header_printed = true;
                        }
                        std::cout << termcolor::reset;
                        std::cout << std::left
                                  << std::fixed << std::setprecision(6) << std::setw(14) << fratio_new
                                  << std::fixed << std::setprecision(6) << std::setw(14) << eratio_new
                                  << std::fixed << std::setprecision(6) << std::setw(14) << fsprial_ratio
                                  << std::fixed << std::setprecision(6) << std::setw(14) << esprial_ratio
                                  << std::fixed << std::setprecision(6) << std::setw(14) << fratio
                                  << std::fixed << std::setprecision(6) << std::setw(14) << eratio
                                  << std::setw(12) << quadquality.get_num_of_complex()
                                  << std::setw(12) << quadquality.get_irregular_vertex_num()
                                  << std::setw(12) << pointer->get_num_of_vertices()
                                  << std::setw(12) << pointer->get_num_of_edges()
                                  << std::setw(12) << pointer->get_num_of_faces()
                                  << std::setw(14) << pointer->get_num_of_boundaries()
                                  << std::setw(8) << pointer->genus()
                                  << '\n';
                    }

                    json_off << "{\n";
                    json_off << "\"submesh_id\": " << submesh_count << ",\n";
                    json_off << "\"FratioN\": " << fratio_new << ",\n";
                    json_off << "\"EratioN\": " << eratio_new << ",\n";
                    json_off << "\"Fspiral\": " << fsprial_ratio << ",\n";
                    json_off << "\"Espiral\": " << esprial_ratio << ",\n";
                    json_off << "\"Fratio\": " << fratio << ",\n";
                    json_off << "\"Eratio\": " << eratio << ",\n";
                    json_off << "\"NumOfComplex\": " << quadquality.get_num_of_complex() << ",\n";
                    json_off << "\"NumOfIrregularVertices\": " << quadquality.get_irregular_vertex_num() << ",\n";
                    json_off << "\"NumOfVertices\": " << pointer->get_num_of_vertices() << ",\n";
                    json_off << "\"NumOfFaces\": " << pointer->get_num_of_faces() << ",\n";
                    json_off << "\"NumOfBoundaries\": " << pointer->get_num_of_boundaries() << ",\n";
                    json_off << "\"SJ\": " << quadquality.get_mean_scaled_jacobian() << ",\n";
                    json_off << "\"Genus\": " << pointer->genus() << '\n';

                    if (submesh_count + 1 < submeshes.size())
                        json_off << "},\n";
                    else
                        json_off << "}\n";

                    submesh_count++;

                    if (pointer != submesh)
                        delete pointer;
                    delete submesh;
                }

                json_off << "]\n";
                json_off.close();
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
