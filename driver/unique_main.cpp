#include <iostream>
#include <filesystem>
#include "cxxopts.hpp"
#include "loopquad.h"
#include "MeshWriter.h"
#include "MeshLoader.h"
#include "looputil.h"
#include "quadquality.h"
#include <list>
// inline std::string GetFileExtension(const std::string &FileName)
// {
//     if (FileName.find_last_of(".") != std::string::npos)
//         return FileName.substr(FileName.find_last_of(".") + 1);
//     return "";
// }

// #define UNIQUE
// #define WRITE_PLY
// #define WRITE_INFO
#ifdef UNIQUE
int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("Unique", "Unique (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                            //
            ("i,input", "input dir", cxxopts::value<std::string>())   //
            ("o,output", "output dir", cxxopts::value<std::string>()) //
            ("h,help", "Print help");                                 //

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }
        namespace fs = std::filesystem;
        auto inputdir = result["i"].as<std::string>();
        if (!fs::exists(inputdir))
        {
            std::cout << "Input directory does not exist!" << std::endl;
            exit(0);
        }
        auto outputdir = result["o"].as<std::string>();
        if (!fs::exists(outputdir))
        {
            fs::create_directories(outputdir);
        }

        std::unordered_map<MySortedTuple<ptrdiff_t, 11, false>, std::vector<TinyVector<double, 3>>> database;

        if (fs::exists("database.bin"))
        {
            std::ifstream databasein("database.bin", std::ios::in | std::ios::binary);
            if (!databasein.is_open())
            {
                std::cout << "Error opening database file!" << std::endl;
                exit(0);
            }
            std::cout << "Reading database..." << std::endl;
            size_t datanum = 0;
            databasein.read(reinterpret_cast<char *>(&datanum), sizeof(datanum));
            for (size_t i = 0; i < datanum; i++)
            {
                MySortedTuple<ptrdiff_t, 11, false> info_tuple;
                databasein.read(reinterpret_cast<char *>(&info_tuple[0]), sizeof(info_tuple[0]) * 11);
                size_t bbox_info_size = 0;
                databasein.read(reinterpret_cast<char *>(&bbox_info_size), sizeof(bbox_info_size));
                for (size_t j = 0; j < bbox_info_size; j++)
                {
                    TinyVector<double, 3> size;
                    databasein.read(reinterpret_cast<char *>(&size[0]), sizeof(double) * 3);
                    database[info_tuple].emplace_back(size);
                }
            }
            databasein.close();
            std::cout << "Database read successfully!" << std::endl;
        }

        double bound_threshold = 0.05; // 0.05; // 0.01; //0.001;
        size_t index_find_count = 0, duplicate_count = 0, file_count = 0;
        // for (const auto &entry : fs::recursive_directory_iterator(inputdir))
        for (const auto &entry : fs::directory_iterator(inputdir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".ply")
            {
                // std::cout << "Processing file: " << entry.path() << std::endl;
                auto file_size = fs::file_size(entry.path());
                if (file_size == 0)
                    continue; // skip small files
                // std::cout << entry.path() << std::endl;
                // MeshLib::Mesh3D<double> *m_pmesh = LoadPLYmesh<double>(entry.path().c_str());
                MeshLib::Mesh3D<double> *m_pmesh = load_mesh<double>(entry.path().string());
                if (!m_pmesh)
                    continue;
                std::vector<MeshLib::Mesh3D<double> *> submeshes, keptsubmeshes;
                mesh_decomposition(m_pmesh, submeshes);

                MeshLib::Mesh3D<double> *merged_mesh = new MeshLib::Mesh3D<double>;
                ptrdiff_t num_vertices = 0;
                bool mesh_modifed = false;
                for (auto submesh : submeshes)
                {
                    if (submesh->get_num_of_faces() < 16) // skip small sub-mesh
                    {
                        mesh_modifed = true;
                        delete submesh;
                        continue;
                    }
                    bool find_duplicate = false;

                    MySortedTuple<ptrdiff_t, 11, false> info_tuple;
                    TinyVector<double, 3> size;
                    get_meshinfo(submesh, info_tuple, size);

                    auto it = database.find(info_tuple);
                    if (it == database.end())
                    {
                        std::vector<TinyVector<double, 3>> mesh_info;
                        mesh_info.push_back(size);
                        database[info_tuple] = mesh_info;
                    }
                    else
                    {
                        index_find_count++;
                        // if (index_find_count % 100 == 0)
                        //     std::cout << "Index find count: " << index_find_count << std::endl;
                        auto &bbox_info = it->second;
                        for (size_t i = 0; i < bbox_info.size(); i++)
                        {
                            const auto &msize = bbox_info[i];
                            auto diff = msize - size;
                            auto max_diff = std::max({fabs(diff[0]), fabs(diff[1]), fabs(diff[2])});
                            auto max_side = std::max({fabs(msize[0]), fabs(msize[1]), fabs(msize[2])});
                            if (max_diff / max_side < bound_threshold)
                            {
                                find_duplicate = true;
                                break;
                            }
                        }
                        if (!find_duplicate)
                        {
                            bbox_info.push_back(size);
                        }
                    }
                    if (find_duplicate)
                    {
                        mesh_modifed = true;
                        duplicate_count++;
                        if (duplicate_count % 1000 == 0)
                            std::cout << "Duplicate count: " << duplicate_count << std::endl;
                        delete submesh;
                        continue;
                    }

                    if (!find_duplicate)
                    {
                        keptsubmeshes.push_back(submesh);
                    }
                    // for (int i = 0; i < submesh->get_num_of_vertices(); i++)
                    // {
                    //     auto vert = submesh->get_vertex(i);
                    //     merged_mesh->insert_vertex(vert->pos);
                    // }

                    // for (int i = 0; i < submesh->get_num_of_faces(); i++)
                    // {
                    //     auto face = submesh->get_face(i);
                    //     std::vector<MeshLib::HE_vert<double> *> face_vert;
                    //     auto he = face->edge;
                    //     do
                    //     {
                    //         face_vert.push_back(merged_mesh->get_vertex(he->vert->id + num_vertices));
                    //         he = he->next;
                    //     } while (he != face->edge);
                    //     merged_mesh->insert_face(face_vert);
                    // }
                    // num_vertices += submesh->get_num_of_vertices();
                    // delete submesh;
                }
                if (mesh_modifed)
                {
                    for (auto submesh : keptsubmeshes)
                    {
                        for (int i = 0; i < submesh->get_num_of_vertices(); i++)
                        {
                            auto vert = submesh->get_vertex(i);
                            merged_mesh->insert_vertex(vert->pos);
                        }

                        for (int i = 0; i < submesh->get_num_of_faces(); i++)
                        {
                            auto face = submesh->get_face(i);
                            std::vector<MeshLib::HE_vert<double> *> face_vert;
                            auto he = face->edge;
                            do
                            {
                                face_vert.push_back(merged_mesh->get_vertex(he->vert->id + num_vertices));
                                he = he->next;
                            } while (he != face->edge);
                            merged_mesh->insert_face(face_vert);
                        }
                        num_vertices += submesh->get_num_of_vertices();
                    }
                }
                for (auto submesh : keptsubmeshes)
                {
                    delete submesh;
                }
                delete m_pmesh;
                std::string output_path = result["o"].as<std::string>() + "/" + entry.path().filename().string();

                if (mesh_modifed)
                {
                    if (merged_mesh->get_num_of_faces() > 0)
                        SavePLYmesh_with_float_storage(merged_mesh, output_path.c_str());
                }
                else
                {
                    fs::copy_file(entry.path(), output_path, fs::copy_options::overwrite_existing);
                }
                // if (merged_mesh->get_num_of_faces() == 0)
                // {
                //     fs::remove(output_path);
                //     // std::cout << "deleted " << output_path << " due to too few faces." << std::endl;
                // }
                // if (merged_mesh->get_num_of_faces() > 16)
                // {
                //     if (mesh_modifed)
                //         SavePLYmesh_with_float_storage(merged_mesh, output_path.c_str());
                //     else
                //         fs::copy_file(entry.path(), output_path, fs::copy_options::overwrite_existing);
                // }
                delete merged_mesh;

                file_count++;
                if (file_count % 1000 == 0)
                    std::cout << "File count: " << file_count << std::endl;
            }
        }

        std::cout << database.size();

        std::ofstream databaseout("database.bin", std::ios::out | std::ios::binary);
        if (databaseout.is_open())
        {
            auto datanum = database.size();
            databaseout.write(reinterpret_cast<const char *>(&datanum), sizeof(datanum));
            for (const auto &element : database)
            {
                const auto &first = element.first;
                const auto &second = element.second;
                databaseout.write(reinterpret_cast<const char *>(&first[0]), sizeof(first[0]) * 11);
                auto bbox_info_size = second.size();
                databaseout.write(reinterpret_cast<const char *>(&bbox_info_size), sizeof(bbox_info_size));
                for (const auto &bbox_info : second)
                {
                    databaseout.write(reinterpret_cast<const char *>(&bbox_info[0]), sizeof(double) * 3);
                }
            }
            databaseout.close();
            std::cout << "Database written successfully!" << std::endl;
        }
    }
    catch (const cxxopts::exceptions::exception &e)
    {
        std::cout << "Error parsing options: " << e.what() << std::endl;
        exit(0);
    }
    return 0;
}
#else
int main(int argc, char **argv)
{
    cxxopts::Options options("Unique", "Unique (author: Yang Liu, Email: yangliu@microsoft.com)");
    options
        .positional_help("[optional args]")
        .show_positional_help()
        .allow_unrecognised_options()
        .add_options()                                            //
        ("i,input", "input dir", cxxopts::value<std::string>())   //
        ("o,output", "output dir", cxxopts::value<std::string>()) //
        ("h,help", "Print help");                                 //

    auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }
    namespace fs = std::filesystem;
    auto inputdir = result["i"].as<std::string>();
    if (!fs::exists(inputdir))
    {
        std::cout << "Input directory does not exist!" << std::endl;
        exit(0);
    }
    auto outputdir = result["o"].as<std::string>();
    if (!fs::exists(outputdir))
    {
        fs::create_directories(outputdir);
    }

#ifdef WRITE_INFO
    auto outfilename = outputdir + "/mesh_info.bin";
    std::cout << "Output file: " << outfilename << std::endl;
    // if (fs::exists(outfilename))
    // {
    //     std::cout << "The directory has been processed!" << std::endl;
    //     exit(0);
    // }
#endif

#ifdef WRITE_INFO
    class info_tuple
    {
    public:
        std::string filename;
        unsigned int submesh_id;
        unsigned int num_vertices;
        unsigned int num_faces;
        unsigned int num_boundaries;
        unsigned int genus;
        unsigned int num_complexes;
        unsigned int num_interior_singularities;
        unsigned int num_boundary_singularities;
        unsigned int max_boundary_degree;
        unsigned int max_interior_degree;
        unsigned int max_num_of_singular_vertices_in_face;
        unsigned int has_interior_edge_connecting_with_boundary;
        float edge_score;
        float face_score;
        float dangle_diff;
        float simple_fl_ratio, simple_el_ratio, fl_sprial_ratio, el_sprial_ratio;
        float complexity;
        float size_ratio;
        float min_area_ratio, min_edge_length;
    };
    std::list<info_tuple> info_list;
#endif
    for (const auto &entry : fs::recursive_directory_iterator(inputdir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".ply")
        {
            auto file_size = fs::file_size(entry.path());
            if (file_size == 0)
                continue; // skip small files
            // std::cout << entry.path() << std::endl;
            // MeshLib::Mesh3D<double> *m_pmesh = LoadPLYmesh<double>(entry.path().string().c_str());
            MeshLib::Mesh3D<double> *m_pmesh = load_mesh<double>(entry.path().string());
            std::vector<MeshLib::Mesh3D<double> *> submeshes;
            mesh_decomposition(m_pmesh, submeshes);
            int count = 0;
            for (auto submesh : submeshes)
            {
                if (submesh->is_quad() == false)
                {
                    std::cout << "non_quad" << std::endl;
                    count++;
                    continue;
                }
                size_t num_interior_singularities = 0, num_boundary_singularities = 0;
                double dangle_diff = 0;
                int max_boundary_degree = 0, max_interior_degree = 0, max_num_of_singular_vertices_in_face = 0;
                int has_interior_edge_connecting_with_boundary = 0;
                has_interior_edge_connecting_with_boundary = skip_quad_mesh(submesh, num_interior_singularities, num_boundary_singularities, dangle_diff,
                                                                            max_boundary_degree, max_interior_degree, max_num_of_singular_vertices_in_face);
                // if (skip_quad_mesh(submesh, num_interior_singularities, num_boundary_singularities, dangle_diff,
                //                    max_boundary_degree, max_interior_degree, max_num_of_singular_vertices_in_face))
                // {
                // std::cout << "Skip mesh: " << entry.path() << ", submesh id: " << count << std::endl;
                // QuadQuality<double> quadquality(submesh, false);
                // auto fratio = quadquality.get_simple_faceloop_ratio(), eratio = quadquality.get_simple_edgeloop_ratio();
                // std::cout << "Face ratio: " << fratio << ", Edge ratio: " << eratio << ", Dangle diff: " << dangle_diff << std::endl;
                // auto fratio_new = quadquality.get_simple_faceloop_ratio_new(), eratio_new = quadquality.get_simple_edgeloop_ratio_new();
                // auto fsprial_ratio = quadquality.get_faceloop_spriality_ratio(), esprial_ratio = quadquality.get_edgeloop_spriality_ratio();
                // std::cout << "fs: " << fratio_new << ", es: " << eratio_new << ", fsprial: " << fsprial_ratio << ", eprial: " << esprial_ratio << std::endl;
                // std::cout << "Processing submesh: " << entry.path() << ", submesh id: " << count << ": skip" << std::endl;
                // count++;
                // continue;
                // }

#ifdef WRITE_PLY
                if (submesh->get_num_of_faces() <= 16 || has_interior_edge_connecting_with_boundary == 1 || max_num_of_singular_vertices_in_face > 2 ||
                    dangle_diff > 60 || max_interior_degree > 8 || max_boundary_degree > 3 || num_interior_singularities == 0)
                {
                    // std::cout << "Processing submesh: " << entry.path() << ", submesh id: " << count << ": skip 2" << std::endl;
                    count++;
                    continue;
                }
#endif
                MySortedTuple<ptrdiff_t, 11, false> my_info_tuple;
                TinyVector<double, 3> size;
                get_meshinfo(submesh, my_info_tuple, size, true);
                auto size_ratio = std::min({size[0], size[1], size[2]}) / (std::max({size[0], size[1], size[2]}) + 1.0e-10);

#ifdef WRITE_PLY
                if (size_ratio <= 0.2) // 0.9
                {
                    // std::cout << "Processing submesh: " << entry.path() << ", submesh id: " << count << ": skip size_ratio" << size_ratio << std::endl;
                    count++;
                    continue;
                }
#endif
                QuadQuality<double> quadquality(submesh, false);
                // std::cout << "remain mesh: " << entry.path() << ", submesh id: " << count << std::endl;
                auto fratio = quadquality.get_simple_faceloop_ratio(), eratio = quadquality.get_simple_edgeloop_ratio();
                // std::cout << "Face ratio: " << fratio << ", Edge ratio: " << eratio << ", Dangle diff: " << dangle_diff << std::endl;
                auto fratio_new = quadquality.get_simple_faceloop_ratio_new(), eratio_new = quadquality.get_simple_edgeloop_ratio_new();
                auto fsprial_ratio = quadquality.get_faceloop_spriality_ratio(), esprial_ratio = quadquality.get_edgeloop_spriality_ratio();
                double max_area_ratio, min_area_ratio, mean_area_ratio, min_edge_length;
                quadquality.get_complex_distribution(max_area_ratio, min_area_ratio, mean_area_ratio, min_edge_length);
#ifdef WRITE_PLY
                if (fratio_new < 0.6 || min_area_ratio < 0.001 || min_edge_length < 0.02)
                {
                    // std::cout << fratio_new << ", " << min_area_ratio << ", " << mean_area_ratio << ", " << min_edge_length << std::endl;
                    // std::cout << "Processing submesh: " << entry.path() << ", submesh id: " << count << ": skip 4" << std::endl;

                    count++;
                    continue;
                }
#endif
                // std::cout << "fs: " << fratio_new << ", es: " << eratio_new << ", fsprial: " << fsprial_ratio << ", eprial: " << esprial_ratio << std::endl;
#ifdef WRITE_INFO
                info_tuple info;
                info.filename = entry.path().string();
                info.submesh_id = (unsigned int)count;
                info.num_vertices = (unsigned int)submesh->get_num_of_vertices();
                info.num_faces = (unsigned int)submesh->get_num_of_faces();
                info.num_boundaries = (unsigned int)submesh->get_num_of_boundaries();
                info.genus = (unsigned int)submesh->genus();
                info.num_complexes = (unsigned int)quadquality.get_num_of_complex();
                info.num_interior_singularities = (unsigned int)num_interior_singularities;
                info.num_boundary_singularities = (unsigned int)num_boundary_singularities;
                info.edge_score = (float)eratio;
                info.face_score = (float)fratio;
                info.dangle_diff = (float)dangle_diff;
                info.simple_fl_ratio = (float)fratio_new;
                info.simple_el_ratio = (float)eratio_new;
                info.fl_sprial_ratio = (float)fsprial_ratio;
                info.el_sprial_ratio = (float)esprial_ratio;
                info.complexity = info.num_complexes / (float)info.num_faces;
                info.max_boundary_degree = (unsigned int)max_boundary_degree;
                info.max_interior_degree = (unsigned int)max_interior_degree;
                info.max_num_of_singular_vertices_in_face = (unsigned int)max_num_of_singular_vertices_in_face;
                info.has_interior_edge_connecting_with_boundary = (unsigned int)has_interior_edge_connecting_with_boundary;
                info.size_ratio = (float)size_ratio;
                info.min_area_ratio = (float)min_area_ratio;
                info.min_edge_length = (float)min_edge_length;
                info_list.push_back(info);
#endif
#ifdef WRITE_PLY
                auto nc = quadquality.get_num_of_complex();
                // nc_interval [0-10, 10-20, 20-30, ..., 90-100, 100+]
                int nc_interval = std::min((int)std::floor(nc / 10.0) * 10, 100);

                int escore_interval = (int)std::floor(eratio * 10);
                int fscore_interval = (int)std::floor(fratio * 10);
                int esprial_interval = (int)std::floor(esprial_ratio * 10);
                int fsprial_interval = (int)std::floor(fsprial_ratio * 10);

                auto nf = submesh->get_num_of_faces();
                int nf_interval = std::min((int)std::floor(nf / 1000.0) * 1000, 10000);

                auto genus = submesh->genus();
                int genus_interval = std::min((int)std::floor(genus / 10.0) * 10, 100);

                auto ns = num_interior_singularities + num_boundary_singularities;
                int ns_interval = std::min((int)std::floor(ns / 20.0) * 20, 200);

                // auto subdir = outputdir + "/ns_" + std::to_string(ns_interval);
                // if (!fs::exists(subdir))
                // {
                //     fs::create_directories(subdir);
                // }
                // Save the submesh to the corresponding directory
                auto filename_without_ext_and_path = entry.path().filename().replace_extension("").string();
                // std::string submesh_filename = subdir + "/" + filename_without_ext_and_path + "_" + std::to_string(count) + ".ply";
                std::string submesh_filename = outputdir + "/" + "ns_" + std::to_string(ns_interval) + "_escore_" + std::to_string(escore_interval) + "_fscore_" + std::to_string(fscore_interval) + "_esprial_" + std::to_string(esprial_interval) + "_fsprial_" + std::to_string(fsprial_interval) + "_nc_" + std::to_string(nc_interval) + "_" + filename_without_ext_and_path + "_" + std::to_string(count) + ".ply";
                if (fs::exists(submesh_filename))
                {
                    count++;
                    continue; // skip if file already exists
                }
                SavePLYmesh_with_float_storage(submesh, submesh_filename.c_str(), &quadquality.get_face_complex_ids());
                std::string subemeshchartedgefilename = outputdir + "/" + "ns_" + std::to_string(ns_interval) + "_escore_" + std::to_string(escore_interval) + "_fscore_" + std::to_string(fscore_interval) + "_esprial_" + std::to_string(esprial_interval) + "_fsprial_" + std::to_string(fsprial_interval) + "_nc_" + std::to_string(nc_interval) + "_" + filename_without_ext_and_path + "_" + std::to_string(count) + "_chartedge.ply";
                SaveChartEdge_as_ply(submesh, subemeshchartedgefilename.c_str(), quadquality.get_face_complex_ids());
                std::string mlpfile = outputdir + "/" + "ns_" + std::to_string(ns_interval) + "_escore_" + std::to_string(escore_interval) + "_fscore_" + std::to_string(fscore_interval) + "_esprial_" + std::to_string(esprial_interval) + "_fsprial_" + std::to_string(fsprial_interval) + "_nc_" + std::to_string(nc_interval) + "_" + filename_without_ext_and_path + "_" + std::to_string(count) + ".mlp";
                // get filename without extension and path
                fs::path submesh_filename_path(submesh_filename);
                auto filename_only = submesh_filename_path.stem().string();
                auto edgefilename_only = subemeshchartedgefilename.substr(0, subemeshchartedgefilename.find_last_of('.'));
                SaveMeshlabMLP(mlpfile, fs::path(submesh_filename).filename().string(), fs::path(subemeshchartedgefilename).filename().string());
#endif
                count++;
            }
            for (auto submesh : submeshes)
                delete submesh;
            delete m_pmesh;
        }
    }

#ifdef WRITE_INFO
    // Save the info_list to a binary file
    std::ofstream info_file(outputdir + "/mesh_info.bin", std::ios::binary);
    if (!info_file.is_open())
    {
        std::cerr << "Error opening info file for writing!" << std::endl;
        return 1;
    }
    else
    {
        size_t info_count = info_list.size();
        info_file.write(reinterpret_cast<const char *>(&info_count), sizeof(info_count));
        for (const auto &info : info_list)
        {
            // writ filename in binary format
            size_t filename_length = info.filename.size();
            info_file.write(reinterpret_cast<const char *>(&filename_length), sizeof(filename_length));
            info_file.write(info.filename.c_str(), filename_length);
            // info_file.write(info.filename.c_str(), info.filename.size() + 1);
            info_file.write(reinterpret_cast<const char *>(&info.submesh_id), sizeof(info.submesh_id));
            info_file.write(reinterpret_cast<const char *>(&info.num_vertices), sizeof(info.num_vertices));
            info_file.write(reinterpret_cast<const char *>(&info.num_faces), sizeof(info.num_faces));
            info_file.write(reinterpret_cast<const char *>(&info.num_boundaries), sizeof(info.num_boundaries));
            info_file.write(reinterpret_cast<const char *>(&info.genus), sizeof(info.genus));
            info_file.write(reinterpret_cast<const char *>(&info.num_complexes), sizeof(info.num_complexes));
            info_file.write(reinterpret_cast<const char *>(&info.num_interior_singularities), sizeof(info.num_interior_singularities));
            info_file.write(reinterpret_cast<const char *>(&info.num_boundary_singularities), sizeof(info.num_boundary_singularities));
            info_file.write(reinterpret_cast<const char *>(&info.max_boundary_degree), sizeof(info.max_boundary_degree));
            info_file.write(reinterpret_cast<const char *>(&info.max_interior_degree), sizeof(info.max_interior_degree));
            info_file.write(reinterpret_cast<const char *>(&info.max_num_of_singular_vertices_in_face), sizeof(info.max_num_of_singular_vertices_in_face));
            info_file.write(reinterpret_cast<const char *>(&info.has_interior_edge_connecting_with_boundary), sizeof(info.has_interior_edge_connecting_with_boundary));

            info_file.write(reinterpret_cast<const char *>(&info.edge_score), sizeof(info.edge_score));
            info_file.write(reinterpret_cast<const char *>(&info.face_score), sizeof(info.face_score));
            info_file.write(reinterpret_cast<const char *>(&info.dangle_diff), sizeof(info.dangle_diff));
            info_file.write(reinterpret_cast<const char *>(&info.simple_fl_ratio), sizeof(info.simple_fl_ratio));
            info_file.write(reinterpret_cast<const char *>(&info.simple_el_ratio), sizeof(info.simple_el_ratio));
            info_file.write(reinterpret_cast<const char *>(&info.fl_sprial_ratio), sizeof(info.fl_sprial_ratio));
            info_file.write(reinterpret_cast<const char *>(&info.el_sprial_ratio), sizeof(info.el_sprial_ratio));
            info_file.write(reinterpret_cast<const char *>(&info.complexity), sizeof(info.complexity));
            info_file.write(reinterpret_cast<const char *>(&info.size_ratio), sizeof(info.size_ratio));
            info_file.write(reinterpret_cast<const char *>(&info.min_area_ratio), sizeof(info.min_area_ratio));
            info_file.write(reinterpret_cast<const char *>(&info.min_edge_length), sizeof(info.min_edge_length));
        }
        info_file.close();
    }
#endif

    return 0;
}
// int main(int argc, char **argv)
// {
//     try
//     {
//         cxxopts::Options options("Unique", "Unique (author: Yang Liu, Email: yangliu@microsoft.com)");
//         options
//             .positional_help("[optional args]")
//             .show_positional_help()
//             .allow_unrecognised_options()
//             .add_options()                                            //
//             ("i,input", "input dir", cxxopts::value<std::string>())   //
//             ("o,output", "output dir", cxxopts::value<std::string>()) //
//             ("h,help", "Print help");                                 //

//         auto result = options.parse(argc, argv);
//         if (result.count("help"))
//         {
//             std::cout << options.help({"", "Group"}) << std::endl;
//             exit(0);
//         }
//         namespace fs = std::filesystem;
//         auto inputdir = result["i"].as<std::string>();
//         if (!fs::exists(inputdir))
//         {
//             std::cout << "Input directory does not exist!" << std::endl;
//             exit(0);
//         }
//         auto outputdir = result["o"].as<std::string>();
//         if (!fs::exists(outputdir))
//         {
//             fs::create_directories(outputdir);
//         }
//         std::vector<std::string> plyfilelist, filenamelist;
//         plyfilelist.reserve(10000), filenamelist.reserve(10000);
//         for (const auto &entry : fs::recursive_directory_iterator(inputdir))
//         {
//             if (entry.is_regular_file() && entry.path().extension() == ".ply")
//             {
//                 plyfilelist.emplace_back(entry.path().string());
//                 filenamelist.emplace_back(entry.path().filename().string());
//             }
//         }
// #pragma omp parallel for schedule(dynamic)
//         for (int i = 0; i < (int)plyfilelist.size(); i++)
//         {
//             std::string output_path = result["o"].as<std::string>() + "/" + filenamelist[i];
//             merge_ply_mesh_enhanced<double>(plyfilelist[i], output_path);
//             // bool suc = merge_ply_mesh<double>(plyfilelist[i], output_path);
//             // if (!suc)
//             // {
//             //     std::filesystem::copy_file(plyfilelist[i], output_path, fs::copy_options::overwrite_existing);
//             // }
//         }
//     }
//     catch (const cxxopts::exceptions::exception &e)
//     {
//         std::cout << "Error parsing options: " << e.what() << std::endl;
//         exit(0);
//     }
//     return 0;
// }
#endif