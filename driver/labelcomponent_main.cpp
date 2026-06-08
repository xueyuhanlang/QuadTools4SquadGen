#include "cxxopts.hpp"
#include <array>
#include <filesystem>
#include <iostream>
#include "libnpy/npz.h"
#include "NonmanifoldProcess.h"
#include "MeshLoader.h"
#include "MeshWriter.h"

int main(int argc, char **argv)
{
    try
    {
        cxxopts::Options options("LabelComponent", "LabelComponent (author: Yang Liu, Email: yangliu@microsoft.com)");
        options
            .positional_help("[optional args]")
            .show_positional_help()
            .allow_unrecognised_options()
            .add_options()                                                               //
            ("i,input", "input filename (*.npz,*.bin)", cxxopts::value<std::string>())   //
            ("o,output", "output filename (*.npz,*.ply)", cxxopts::value<std::string>()) //
            ("h,help", "Print help");                                                    //

        auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        if (result.count("i") == 0)
        {
            std::cout << "No input file!" << std::endl;
            std::cout << "Please use -i option" << std::endl;
            exit(0);
        }
        if (result.count("o") == 0)
        {
            std::cout << "No output file!" << std::endl;
            std::cout << "Please use -o option" << std::endl;
        }

        auto inputfile = result["i"].as<std::string>();
        auto outputfile = result["o"].as<std::string>();

        if (!std::filesystem::exists(inputfile))
        {
            std::cout << "Input file does not exist!" << std::endl;
            exit(0);
        }

        auto ext = std::filesystem::path(inputfile).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext != ".npz" && ext != ".bin")
        {
            std::cout << "Unsupport format!" << std::endl;
            exit(0);
        }
        int num_objects = 0;
        size_t total_vertices = 0, total_faces = 0;
        std::vector<std::vector<std::vector<size_t>>> all_face_indices;
        std::vector<std::vector<TinyVector<double, 3>>> all_vertices;

        if (ext == ".npz")
        {
            npy::inpzstream input(inputfile);
            if (!input.is_open())
            {
                std::cerr << "Cannot open the file!" << std::endl;
                exit(0);
            }
            num_objects = (int)input.read<std::int64_t>("num_objects.npy")(0);
            // std::cout << "There are " << num_objects << " objects in the input file." << std::endl;
            all_face_indices.resize(num_objects);
            all_vertices.resize(num_objects);
            for (auto i = 0; i < num_objects; ++i)
            {
                auto vertices_tensor = input.read<float>("vertices_" + std::to_string(i) + ".npy");
                auto faces_tensor = input.read<int32_t>("faces_" + std::to_string(i) + ".npy");
                // std::cout << vertices_tensor.shape()[0] << vertices_tensor.shape()[1] << std::endl;
                // std::cout << faces_tensor.shape()[0] << faces_tensor.shape()[1] << std::endl;
                total_vertices += vertices_tensor.shape()[0];
                total_faces += faces_tensor.shape()[0];
                all_face_indices[i].resize(faces_tensor.shape()[0]);
                all_vertices[i].reserve(vertices_tensor.shape()[0]);
                for (auto v = 0; v < vertices_tensor.shape()[0]; ++v)
                {
                    all_vertices[i].emplace_back(TinyVector<double, 3>(vertices_tensor(v, 0), vertices_tensor(v, 1), vertices_tensor(v, 2)));
                }
                for (auto f = 0; f < faces_tensor.shape()[0]; ++f)
                {
                    all_face_indices[i][f].push_back(faces_tensor(f, 0));
                    all_face_indices[i][f].push_back(faces_tensor(f, 1));
                    all_face_indices[i][f].push_back(faces_tensor(f, 2));
                }
            }

            input.close();
        }
        else
        {
            // python code for writing bin file
            // with open(temp_output_name, "wb") as f:
            //     np.array([count], dtype=np.int32).tofile(f)
            //     for i in range(count):
            //         np.array(save_dict[f"vertices_{i}"].shape[0], dtype=np.int32).tofile(f)
            //         np.array(save_dict[f"faces_{i}"].shape[0], dtype=np.int32).tofile(f)
            //     for i in range(count):
            //         save_dict[f"vertices_{i}"].tofile(f)
            //         save_dict[f"faces_{i}"].tofile(f)

            // so the reading code should be like this
            std::ifstream input(inputfile, std::ios::binary);
            if (!input.is_open())
            {
                std::cerr << "Cannot open the file!" << std::endl;
                exit(0);
            }
            input.read((char *)&num_objects, sizeof(int32_t));
            all_face_indices.resize(num_objects);
            all_vertices.resize(num_objects);
            std::vector<int32_t> vertex_counts(num_objects), face_counts(num_objects);
            for (auto i = 0; i < num_objects; ++i)
            {
                input.read((char *)&vertex_counts[i], sizeof(int32_t));
                input.read((char *)&face_counts[i], sizeof(int32_t));
                total_vertices += vertex_counts[i];
                total_faces += face_counts[i];
            }
            for (auto i = 0; i < num_objects; ++i)
            {
                all_vertices[i].resize(vertex_counts[i]);
                all_face_indices[i].resize(face_counts[i]);
                float p[3];
                for (auto v = 0; v < vertex_counts[i]; ++v)
                {
                    input.read(reinterpret_cast<char *>(p), 3 * sizeof(float));
                    all_vertices[i][v] = TinyVector<double, 3>(p[0], p[1], p[2]);
                }
                for (auto f = 0; f < face_counts[i]; ++f)
                {
                    all_face_indices[i][f].resize(3);
                    int32_t temp_face[3];
                    input.read((char *)temp_face, 3 * sizeof(int32_t));
                    all_face_indices[i][f][0] = static_cast<size_t>(temp_face[0]);
                    all_face_indices[i][f][1] = static_cast<size_t>(temp_face[1]);
                    all_face_indices[i][f][2] = static_cast<size_t>(temp_face[2]);
                }
            }
        }

        auto backup_vertices = all_vertices;
        auto backup_faces = all_face_indices;

        size_t submesh_count = 0;
        std::vector<size_t> face_submesh_ids, total_face_submesh_ids;
        total_face_submesh_ids.reserve(total_faces);
        for (auto i = 0; i < num_objects; ++i)
        {
            size_t original_face_count = all_face_indices[i].size();
            size_t original_vertex_count = all_vertices[i].size();
            bool nonmanifold_issue = false;
            // save_mesh(all_vertices[i], all_face_indices[i], "mesh_" + std::to_string(i) + ".ply");
            merge_boundary_vertices(all_vertices[i], all_face_indices[i], nonmanifold_issue, true, false);
            face_submesh_ids.resize(0);
            submesh_count = label_connected_components(all_vertices[i], all_face_indices[i], face_submesh_ids, submesh_count);
            total_face_submesh_ids.insert(total_face_submesh_ids.end(), face_submesh_ids.begin(), face_submesh_ids.end());
            if (all_face_indices[i].size() != original_face_count)
            {
                std::cout << "face count changes: " << original_face_count << " -> " << all_face_indices[i].size() << std::endl;
                std::cout << inputfile << std::endl;
                return 0;
            }
        }
        // std::cout << "Total component: " << submesh_count << std::endl;
        auto out_ext = std::filesystem::path(outputfile).extension().string();
        std::transform(out_ext.begin(), out_ext.end(), out_ext.begin(), ::tolower);
        if (out_ext == ".npz")
        {
            npy::onpzstream output(outputfile);
            std::vector<size_t> shape({total_face_submesh_ids.size(), 1});
            npy::tensor<std::uint32_t> output_tensor(shape);
#pragma omp parallel for
            for (int i = 0; i < (int)total_face_submesh_ids.size(); ++i)
            {
                output_tensor(i, 0) = static_cast<std::uint32_t>(total_face_submesh_ids[i]);
            }
            output.write("face2component", output_tensor);
            output.close();
        }
        else if (out_ext == ".ply")
        {
            std::vector<TinyVector<double, 3>> total_vertices;
            std::vector<std::vector<size_t>> total_faces;
            total_vertices.reserve(total_vertices.size());
            total_faces.reserve(total_faces.size());
            std::vector<size_t> new_face;
            new_face.reserve(3);
            for (auto i = 0; i < num_objects; ++i)
            {
                size_t vertex_offset = total_vertices.size();
                total_vertices.insert(total_vertices.end(), backup_vertices[i].begin(), backup_vertices[i].end());
                for (const auto &face : backup_faces[i])
                {
                    new_face.resize(0);
                    for (const auto v_id : face)
                    {
                        new_face.push_back(v_id + vertex_offset);
                    }
                    total_faces.emplace_back(new_face);
                }
            }
            SavePLYMesh_with_color(outputfile, total_vertices, total_faces, 0, &total_face_submesh_ids);
        }
        else
        {
            std::cout << "Only *.npz and *.ply files are supported!" << std::endl;
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
