#define _CRT_SECURE_NO_WARNINGS

#include "MeshLoader.h"

#include <fstream>
#include <algorithm>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_USE_RAPIDJSON
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION
#include "tiny_gltf.h"
#include "happly.h"
#include "ufbx.h"
#include "NonmanifoldProcess.h"
////////////////////////////////////////////////
template <typename Real>
void mesh_load_interface(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, bool nonmanifold_input)
{
    // get the file extension
    size_t lastindex = filename.find_last_of(".");
    std::string ext = filename.substr(lastindex + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                   { return std::tolower(c); }); //
    if (ext == "ply")
    {
        mesh_load_ply(filename, vertices, face_indices);
    }
    else if (ext == "obj")
    {
        mesh_load_obj(filename, vertices, face_indices);
    }
    else if (ext == "off")
    {
        mesh_load_off(filename, vertices, face_indices);
    }
    else if (ext == "stl")
    {
        mesh_load_stl(filename, vertices, face_indices);
    }
    else if (ext == "gltf" || ext == "glb")
    {
        mesh_load_glb(filename, vertices, face_indices);
    }
    else if (ext == "vtk")
    {
        mesh_load_vtk(filename, vertices, face_indices);
    }
    else if (ext == "vtp")
    {
        mesh_load_vtp(filename, vertices, face_indices);
    }
    else if (ext == "wrl")
    {
        mesh_load_wrl(filename, vertices, face_indices);
    }
    else if (ext == "fbx")
    {
        mesh_load_fbx(filename, vertices, face_indices);
    }
    else
    {
        throw std::runtime_error("Unsupported file format: " + ext);
    }

    if (nonmanifold_input)
    {
        nonmanifold_merge(vertices, face_indices);
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_stl(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.resize(0), face_indices.resize(0);

    try
    {
        std::ifstream fin(filename);
        std::string head;
        fin >> head;
        if (head != "solid")
        {
            fin.close();
            FILE *fp = fopen(filename.c_str(), "rb");
            uint8_t header[80];
            uint32_t nf;
            uint16_t attr;
            float v[3];
            TinyVector<double, 3> V;
            auto __res = fread(header, sizeof(uint8_t), 80, fp);
            __res = fread(&nf, sizeof(uint32_t), 1, fp);
            vertices.resize(nf * 3);
            for (uint32_t id = 0; id < nf; id++)
            {
                __res = fread(v, sizeof(float), 3, fp); // normal
                for (int i = 0; i < 3; i++)
                {
                    __res = fread(v, sizeof(float), 3, fp);
                    V[0] = (double)v[0];
                    V[1] = (double)v[1];
                    V[2] = (double)v[2];
                    vertices.emplace_back(TinyVector<Real, 3>(V[0], V[1], V[2]));
                }
                __res = fread(&attr, sizeof(uint16_t), 1, fp);
            }
            fclose(fp);
            face_indices.resize(vertices.size() / 3);
            for (size_t i = 0; i < vertices.size(); i += 3)
                face_indices[i / 3] = {(uint32_t)i, (uint32_t)(i + 1), (uint32_t)(i + 2)};
        }
        else
        {
            TinyVector<double, 3> V;
            std::string name;
            fin >> name;

            fin >> name;
            bool startface = false;
            while (name != "endsolid")
            {
                if (name == "facet")
                {
                    fin >> name; // normal
                    fin >> V[0] >> V[1] >> V[2];
                }
                else if (name == "outer")
                {
                    fin >> name; // loop
                    startface = true;
                }
                else if (name == "vertex")
                {
                    fin >> V[0] >> V[1] >> V[2];
                    vertices.emplace_back(TinyVector<Real, 3>(V[0], V[1], V[2]));
                }
                else if (name == "endloop")
                {
                    startface = false;
                }
                // else if(name == "end_facet")
                //{
                // }
                fin >> name;
            }
            face_indices.resize(vertices.size() / 3);
            for (size_t i = 0; i < vertices.size(); i += 3)
                face_indices[i / 3] = {(uint32_t)i, (uint32_t)(i + 1), (uint32_t)(i + 2)};
        }
        fin.close();
    }
    catch (const std::exception &e)
    {
        vertices.resize(0), face_indices.resize(0);
        std::cerr << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_ply(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    try
    {
        happly::PLYData plyIn(filename);
        std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
        face_indices = plyIn.getFaceIndices<size_t>();
        vertices.resize(vPos.size());
        for (ptrdiff_t i = 0; i < (ptrdiff_t)vPos.size(); i++)
        {
            vertices[i][0] = (Real)vPos[i][0], vertices[i][1] = (Real)vPos[i][1], vertices[i][2] = (Real)vPos[i][2];
        }
    }
    catch (const std::exception &e)
    {
        vertices.resize(0), face_indices.resize(0);
        std::cerr << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_off(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.clear();
    face_indices.clear();

    try
    {
        // Read entire file into memory for faster parsing
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::string buffer(size, '\0');
        if (!file.read(buffer.data(), size))
        {
            throw std::runtime_error("Cannot read file: " + filename);
        }

        // Parse buffer
        const char *data = buffer.data();
        const char *end = data + size;

        // Skip whitespace and find first line
        while (data < end && (*data == ' ' || *data == '\t' || *data == '\n' || *data == '\r'))
        {
            ++data;
        }

        // Check for OFF header
        if (data + 3 > end || strncmp(data, "OFF", 3) != 0)
        {
            throw std::runtime_error("Invalid OFF file format - missing OFF header");
        }
        data += 3;

        // Skip to next line
        while (data < end && *data != '\n' && *data != '\r')
            ++data;
        while (data < end && (*data == '\n' || *data == '\r'))
            ++data;

        // Parse counts
        int vsize, fsize, esize;
        char *next;

        // Skip whitespace
        while (data < end && (*data == ' ' || *data == '\t'))
            ++data;
        vsize = static_cast<int>(std::strtol(data, &next, 10));
        data = next;

        while (data < end && (*data == ' ' || *data == '\t'))
            ++data;
        fsize = static_cast<int>(std::strtol(data, &next, 10));
        data = next;

        while (data < end && (*data == ' ' || *data == '\t'))
            ++data;
        esize = static_cast<int>(std::strtol(data, &next, 10));
        data = next;

        // Skip to next line
        while (data < end && *data != '\n' && *data != '\r')
            ++data;
        while (data < end && (*data == '\n' || *data == '\r'))
            ++data;

        if (vsize <= 0 || fsize <= 0)
        {
            throw std::runtime_error("Invalid vertex or face count in OFF file");
        }

        // Reserve space
        vertices.reserve(vsize);
        face_indices.reserve(fsize);

        // Parse vertices
        for (int i = 0; i < vsize; ++i)
        {
            TinyVector<Real, 3> vertex;

            for (int j = 0; j < 3; ++j)
            {
                // Skip whitespace
                while (data < end && (*data == ' ' || *data == '\t' || *data == '\n' || *data == '\r'))
                {
                    ++data;
                }

                if (data >= end)
                {
                    throw std::runtime_error("Unexpected end of file while parsing vertices");
                }

                // Parse coordinate
                vertex[j] = static_cast<Real>(std::strtod(data, &next));
                data = next;
            }

            vertices.emplace_back(vertex);
        }

        // Parse faces
        std::vector<size_t> face;
        face.reserve(8); // Reserve for typical polygon sizes

        for (int i = 0; i < fsize; ++i)
        {
            // Skip whitespace
            while (data < end && (*data == ' ' || *data == '\t' || *data == '\n' || *data == '\r'))
            {
                ++data;
            }

            if (data >= end)
            {
                throw std::runtime_error("Unexpected end of file while parsing faces");
            }

            // Parse valence
            int valence = static_cast<int>(std::strtol(data, &next, 10));
            data = next;

            if (valence < 3)
            {
                throw std::runtime_error("Invalid face with less than 3 vertices");
            }

            face.clear();
            face.reserve(valence);

            // Parse face indices
            for (int j = 0; j < valence; ++j)
            {
                // Skip whitespace
                while (data < end && (*data == ' ' || *data == '\t'))
                {
                    ++data;
                }

                if (data >= end)
                {
                    throw std::runtime_error("Unexpected end of file while parsing face indices");
                }

                int idx = static_cast<int>(std::strtol(data, &next, 10));
                data = next;

                if (idx < 0 || idx >= vsize)
                {
                    throw std::runtime_error("Invalid vertex index in face");
                }

                face.emplace_back(static_cast<size_t>(idx));
            }

            face_indices.emplace_back(face);
        }
    }
    catch (const std::exception &e)
    {
        vertices.clear();
        face_indices.clear();
        std::cerr << "Error loading OFF file: " << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_obj(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.clear();
    face_indices.clear();

    try
    {
        // Read entire file into memory
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::string buffer(size, '\0');
        if (!file.read(buffer.data(), size))
        {
            throw std::runtime_error("Cannot read file: " + filename);
        }

        // Reserve space (rough estimates)
        vertices.reserve(size / 50);     // Rough estimate: ~50 chars per vertex line
        face_indices.reserve(size / 40); // Rough estimate: ~40 chars per face line

        // Parse buffer
        const char *data = buffer.data();
        const char *end = data + size;
        const char *line_start = data;

        std::vector<size_t> face;
        face.reserve(8);

        while (line_start < end)
        {
            // Find line end
            const char *line_end = line_start;
            while (line_end < end && *line_end != '\n' && *line_end != '\r')
            {
                ++line_end;
            }

            if (line_end > line_start)
            {
                if (line_start[0] == 'v' && line_start[1] == ' ')
                {
                    // Parse vertex
                    const char *p = line_start + 2;
                    TinyVector<Real, 3> vertex;

                    for (int i = 0; i < 3; ++i)
                    {
                        // Skip whitespace
                        while (p < line_end && (*p == ' ' || *p == '\t'))
                            ++p;

                        // Parse number
                        char *next;
                        vertex[i] = static_cast<Real>(std::strtod(p, &next));
                        p = next;
                    }
                    vertices.emplace_back(vertex);
                }
                else if (line_start[0] == 'f' && line_start[1] == ' ')
                {
                    // Parse face
                    face.clear();
                    const char *p = line_start + 2;

                    while (p < line_end)
                    {
                        // Skip whitespace
                        while (p < line_end && (*p == ' ' || *p == '\t'))
                            ++p;
                        if (p >= line_end)
                            break;

                        // Parse vertex index (before '/' or space)
                        char *next;
                        long idx = std::strtol(p, &next, 10) - 1; // OBJ is 1-indexed
                        if (idx >= 0)
                        {
                            face.emplace_back(static_cast<size_t>(idx));
                        }

                        // Skip to next number (skip texture/normal indices)
                        p = next;
                        while (p < line_end && *p != ' ' && *p != '\t')
                            ++p;
                    }

                    if (face.size() >= 3)
                    {
                        face_indices.emplace_back(face);
                    }
                }
            }

            // Move to next line
            line_start = line_end;
            while (line_start < end && (*line_start == '\n' || *line_start == '\r'))
            {
                ++line_start;
            }
        }
    }
    catch (const std::exception &e)
    {
        vertices.clear();
        face_indices.clear();
        std::cerr << "Error loading OBJ: " << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_glb(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    struct Mat4
    {
        std::array<double, 16> m; // glTF uses double in TinyGLTF

        static Mat4 Identity()
        {
            Mat4 I;
            I.m = {1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1};
            return I;
        }
    };

    class GLTFTransform
    {
    public:
        GLTFTransform(tinygltf::Model &model)
        {
            // Check scene_id validity
            for (size_t scene_id = 0; scene_id < model.scenes.size(); ++scene_id)
            {
                const auto &scene = model.scenes[scene_id];
                std::vector<Mat4> nodeWorld(scene.nodes.size(), Mat4::Identity());
                for (int rootNode : scene.nodes)
                {
                    traverseNode(model, rootNode, Mat4::Identity(), nodeWorld);
                }
                global_scene_node_transform_matrices.emplace_back(nodeWorld);
            }
        }

        const std::vector<std::vector<Mat4>> &GetGlobalTransformations() const
        {
            return global_scene_node_transform_matrices;
        }

    protected:
        Mat4 multiply(const Mat4 &A, const Mat4 &B)
        {
            Mat4 R;
            R.m.fill(0.0);
            for (int r = 0; r < 4; ++r)
            {
                for (int c = 0; c < 4; ++c)
                {
                    for (int k = 0; k < 4; ++k)
                    {
                        R.m[r * 4 + c] += A.m[r * 4 + k] * B.m[k * 4 + c];
                    }
                }
            }
            return R;
        }

        Mat4 translate(const std::array<double, 3> &t)
        {
            Mat4 M = Mat4::Identity();
            M.m[12] = t[0];
            M.m[13] = t[1];
            M.m[14] = t[2];
            return M;
        }

        Mat4 scaleM(const std::array<double, 3> &s)
        {
            Mat4 M = Mat4::Identity();
            M.m[0] = s[0];
            M.m[5] = s[1];
            M.m[10] = s[2];
            return M;
        }

        // Quaternion (x,y,z,w) to matrix (right-handed, column vectors, row-major storage)
        Mat4 quatToMat4(const std::array<double, 4> &q)
        {
            double x = q[0], y = q[1], z = q[2], w = q[3];
            double xx = x * x, yy = y * y, zz = z * z;
            double xy = x * y, xz = x * z, yz = y * z;
            double wx = w * x, wy = w * y, wz = w * z;

            Mat4 M = Mat4::Identity();
            M.m[0] = 1.0 - 2.0 * (yy + zz);
            M.m[1] = 2.0 * (xy + wz);
            M.m[2] = 2.0 * (xz - wy);

            M.m[4] = 2.0 * (xy - wz);
            M.m[5] = 1.0 - 2.0 * (xx + zz);
            M.m[6] = 2.0 * (yz + wx);

            M.m[8] = 2.0 * (xz + wy);
            M.m[9] = 2.0 * (yz - wx);
            M.m[10] = 1.0 - 2.0 * (xx + yy);
            return M;
        }

        // Build local matrix for a node (matrix or T*R*S as per glTF spec).
        Mat4 localNodeMatrix(const tinygltf::Node &node)
        {
            if (!node.matrix.empty() && node.matrix.size() == 16)
            {
                Mat4 M;
                for (int i = 0; i < 16; ++i)
                    M.m[i] = node.matrix[i];
                return M;
            }
            std::array<double, 3> t = {0, 0, 0};
            std::array<double, 3> s = {1, 1, 1};
            std::array<double, 4> r = {0, 0, 0, 1};

            if (node.translation.size() == 3)
            {
                t = {node.translation[0], node.translation[1], node.translation[2]};
            }
            if (node.scale.size() == 3)
            {
                s = {node.scale[0], node.scale[1], node.scale[2]};
            }
            if (node.rotation.size() == 4)
            {
                r = {node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]};
            }

            Mat4 T = translate(t);
            Mat4 R = quatToMat4(r);
            Mat4 S = scaleM(s);

            // glTF defines local transform as T * R * S (applied in that order)
            return multiply(multiply(T, R), S);
        }

        // Recursively traverse the scene graph and compute world transforms.
        void traverseNode(const tinygltf::Model &model,
                          int nodeIndex,
                          const Mat4 &parentWorld,
                          std::vector<Mat4> &nodeWorld)
        {
            const tinygltf::Node &node = model.nodes[nodeIndex];
            Mat4 local = localNodeMatrix(node);
            Mat4 world = multiply(parentWorld, local);
            nodeWorld[nodeIndex] = world;

            for (int child : node.children)
            {
                traverseNode(model, child, world, nodeWorld);
            }
        }

    protected:
        std::vector<std::vector<Mat4>> global_scene_node_transform_matrices;
    };

    vertices.resize(0);
    face_indices.resize(0);

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    bool ret = false;
    if (ext == "glb")
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    else if (ext == "gltf")
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    else
        return;

    if (!warn.empty())
        std::cout << "Warn: " << warn << std::endl;

    if (!err.empty())
        std::cout << "Err: " << err << std::endl;

    if (!ret)
    {
        std::cout << "Failed to parse glTF!" << std::endl;
        return;
    }

    size_t total_num_vertices = 0, total_num_faces = 0;

    for (const tinygltf::Mesh &mesh : model.meshes)
    {
        for (const tinygltf::Primitive &primitive : mesh.primitives)
        {
            for (const auto &attrib : primitive.attributes)
            {
                if (attrib.first.compare("POSITION") != 0)
                    continue;
                const tinygltf::Accessor &positionAccessor = model.accessors[attrib.second];
                auto numPositions = static_cast<size_t>(positionAccessor.count);
                if (primitive.indices >= 0)
                {
                    const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
                    auto numFacets = static_cast<size_t>(indexAccessor.count / 3);
                    if (numFacets > 0)
                    {
                        total_num_vertices += numPositions;
                        total_num_faces += static_cast<size_t>(indexAccessor.count / 3);
                    }
                }
            }
        }
    }

    vertices.reserve(total_num_vertices), face_indices.reserve(total_num_faces);

    GLTFTransform gltfTransform(model);
    const auto &global_transforms = gltfTransform.GetGlobalTransformations();

    for (size_t scene_idx = 0; scene_idx < model.scenes.size(); scene_idx++)
    {
        const tinygltf::Scene &scene = model.scenes[scene_idx];
        for (size_t node_idx = 0; node_idx < scene.nodes.size(); node_idx++)
        {
            const tinygltf::Node &node = model.nodes[scene.nodes[node_idx]];
            if (node.mesh < 0)
                continue;
            const Mat4 &nodeWorld = global_transforms[scene_idx][scene.nodes[node_idx]];
            const tinygltf::Mesh &mesh = model.meshes[node.mesh];
            for (const tinygltf::Primitive &primitive : mesh.primitives)
            {
                // Access POSITION attribute
                auto posAttrIt = primitive.attributes.find("POSITION");
                if (posAttrIt == primitive.attributes.end())
                    continue;
                const tinygltf::Accessor &positionAccessor = model.accessors[posAttrIt->second];
                const tinygltf::BufferView &positionBufferView = model.bufferViews[positionAccessor.bufferView];
                const tinygltf::Buffer &positionBuffer = model.buffers[positionBufferView.buffer];
                const float *positions = reinterpret_cast<const float *>(&positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset]);
                int numPositions = static_cast<int>(positionAccessor.count);

                // Access indices
                if (primitive.indices < 0)
                    continue;
                const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];
                const auto componentType = indexAccessor.componentType;

                int numFacets = static_cast<int>(indexAccessor.count / 3);
                if (numFacets <= 0)
                    continue;

                size_t vertexStartIndex = vertices.size();

                // Transform and store vertices
                for (int i = 0; i < numPositions; i++)
                {
                    double x = positions[3 * i + 0];
                    double y = positions[3 * i + 1];
                    double z = positions[3 * i + 2];

                    // Apply world transformation
                    double tx = nodeWorld.m[0] * x + nodeWorld.m[4] * y + nodeWorld.m[8] * z + nodeWorld.m[12];
                    double ty = nodeWorld.m[1] * x + nodeWorld.m[5] * y + nodeWorld.m[9] * z + nodeWorld.m[13];
                    double tz = nodeWorld.m[2] * x + nodeWorld.m[6] * y + nodeWorld.m[10] * z + nodeWorld.m[14];

                    vertices.emplace_back(TinyVector<Real, 3>((Real)tx, (Real)ty, (Real)tz));
                }

                // Read and store face indices
                // Map component type to corresponding unsigned integer type
                auto add_faces = [&](auto indices)
                {
                    for (size_t i = 0; i < 3 * numFacets; i += 3)
                        face_indices.push_back({vertexStartIndex + indices[i], vertexStartIndex + indices[i + 1], vertexStartIndex + indices[i + 2]});
                };
                if (componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT)
                    add_faces(reinterpret_cast<const uint16_t *>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]));
                else if (componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT)
                    add_faces(reinterpret_cast<const uint32_t *>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]));
                else if (componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE)
                    add_faces(reinterpret_cast<const uint8_t *>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]));
                else
                    std::cerr << "Unsupported index component type" << std::endl;
            }
        }
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_vtk(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.resize(0);
    face_indices.resize(0);
    try
    {
        std::ifstream fin(filename);
        std::string line;
        // skip header
        for (int i = 0; i < 4; i++)
            std::getline(fin, line);
        // read points
        size_t num_points = 0;
        fin >> line >> num_points >> line;
        vertices.resize(num_points);
        for (size_t i = 0; i < num_points; i++)
        {
            fin >> vertices[i][0] >> vertices[i][1] >> vertices[i][2];
        }
        // read faces
        size_t num_faces = 0;
        fin >> line >> num_faces >> line;
        face_indices.resize(num_faces);
        for (size_t i = 0; i < num_faces; i++)
        {
            size_t valence = 0;
            fin >> valence;
            face_indices[i].resize(valence);
            for (size_t j = 0; j < valence; j++)
            {
                fin >> face_indices[i][j];
            }
        }
        fin.close();
    }
    catch (const std::exception &e)
    {
        vertices.resize(0), face_indices.resize(0);
        std::cerr << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_vtp(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.resize(0);
    face_indices.resize(0);
    try
    {
        std::ifstream fin(filename);
        if (!fin.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::string line;
        // skip header
        while (std::getline(fin, line))
        {
            if (line.find("<Points>") != std::string::npos)
                break;
        }
        while (std::getline(fin, line))
        {
            if (line.find("<DataArray") != std::string::npos && line.find("Float") != std::string::npos)
                break;
        }
        // read points
        std::vector<Real> point_data;
        while (std::getline(fin, line))
        {
            if (line.find("</DataArray>") != std::string::npos)
                break;
            std::istringstream iss(line);
            Real val;
            while (iss >> val)
            {
                point_data.push_back(val);
            }
        }
        size_t num_points = point_data.size() / 3;
        vertices.resize(num_points);
        for (size_t i = 0; i < num_points; i++)
        {
            vertices[i][0] = point_data[3 * i];
            vertices[i][1] = point_data[3 * i + 1];
            vertices[i][2] = point_data[3 * i + 2];
        }
        // read faces
        while (std::getline(fin, line))
        {
            if (line.find("<Polys>") != std::string::npos)
                break;
        }
        // read connectivity
        while (std::getline(fin, line))
        {
            if (line.find("<DataArray") != std::string::npos && line.find("Int") != std::string::npos)
                break;
        }
        std::vector<size_t> connectivity;
        while (std::getline(fin, line))
        {
            if (line.find("</DataArray>") != std::string::npos)
                break;
            std::istringstream iss(line);
            size_t idx;
            while (iss >> idx)
            {
                connectivity.push_back(idx);
            }
        }
        // read offsets
        while (std::getline(fin, line))
        {
            if (line.find("<DataArray") != std::string::npos && line.find("Int") != std::string::npos)
                break;
        }
        std::vector<size_t> offsets;
        while (std::getline(fin, line))
        {
            if (line.find("</DataArray>") != std::string::npos)
                break;
            std::istringstream iss(line);
            size_t offset;
            while (iss >> offset)
            {
                offsets.push_back(offset);
            }
        }
        // construct face indices
        if (!connectivity.empty() && !offsets.empty())
        {
            size_t start = 0;
            for (size_t i = 0; i < offsets.size(); i++)
            {
                size_t end = offsets[i];
                if (end > connectivity.size())
                {
                    throw std::runtime_error("Offset value exceeds connectivity size.");
                }
                std::vector<size_t> face;
                for (size_t j = start; j < end; j++)
                {
                    face.push_back(connectivity[j]);
                }
                face_indices.push_back(face);
                start = end;
            }
        }
        fin.close();
    }
    catch (const std::exception &e)
    {
        vertices.resize(0), face_indices.resize(0);
        std::cerr << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_wrl(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.resize(0), face_indices.resize(0);
    try
    {
        std::ifstream fin(filename);
        std::string line;
        // read until we find "point ["
        while (std::getline(fin, line))
        {
            if (line.find("point [") != std::string::npos)
                break;
        }
        // read vertices
        TinyVector<Real, 3> v;
        while (std::getline(fin, line))
        {
            if (line.find("]") != std::string::npos)
                break;
            std::istringstream iss(line);
            iss >> v[0] >> v[1] >> v[2];
            vertices.emplace_back(v);
        }
        // read until we find "coordIndex ["
        while (std::getline(fin, line))
        {
            if (line.find("coordIndex [") != std::string::npos)
                break;
        }
        // read faces
        std::vector<size_t> face;
        while (std::getline(fin, line))
        {
            if (line.find("]") != std::string::npos)
                break;
            face.resize(0);
            std::istringstream iss(line);
            size_t idx;
            while (iss >> idx)
            {
                if (idx == (size_t)-1)
                    break;
                face.push_back(idx);
            }
            if (!face.empty())
                face_indices.emplace_back(face);
        }
        fin.close();
    }
    catch (const std::exception &e)
    {
        vertices.resize(0), face_indices.resize(0);
        std::cerr << e.what() << '\n';
    }
}
////////////////////////////////////////////////
template <typename Real>
void mesh_load_fbx(const std::string &filename, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.resize(0), face_indices.resize(0);

    ufbx_scene *scene = ufbx_load_file(filename.c_str(), nullptr, nullptr);
    if (!scene)
    {
        std::cerr << "Failed to load FBX file: " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < scene->nodes.count; i++)
    {
        ufbx_node *node = scene->nodes.data[i];
        ufbx_mesh *mesh = node->mesh;
        if (!mesh)
            continue;
        std::vector<ufbx_vec3> corners;
        std::vector<uint32_t> face_offsets;
        corners.reserve(mesh->faces.count * 4); // assuming average valence of 4

        for (size_t f = 0; f < mesh->faces.count; f++)
        {
            ufbx_face face = mesh->faces.data[f];
            face_offsets.push_back((uint32_t)corners.size());
            for (size_t j = 0; j < face.num_indices; j++)
            {
                size_t vi = face.index_begin + j;
                auto v = ufbx_get_vertex_vec3(&mesh->vertex_position, vi);
                corners.emplace_back(v);
            }
        }
        // Weld corners into shared vertices
        ufbx_vertex_stream stream;
        stream.data = corners.data();
        stream.vertex_count = (uint32_t)corners.size();
        stream.vertex_size = (uint32_t)sizeof(ufbx_vec3);
        std::vector<uint32_t> corner_to_vertex(corners.size());
        auto num_vertices =
            ufbx_generate_indices(&stream, 1,
                                  corner_to_vertex.data(), corner_to_vertex.size(),
                                  nullptr, nullptr);
        corners.resize(num_vertices);
        auto start_pos = vertices.size();
        for (size_t i = 0; i < corners.size(); i++)
        {
            // apply geometric transform
            ufbx_vec3 v = ufbx_transform_position(&node->geometry_to_world, corners[i]);
            vertices.emplace_back(TinyVector<Real, 3>((Real)v.x, (Real)v.y, (Real)v.z));
        }
        for (size_t f = 0; f < mesh->faces.count; f++)
        {
            std::vector<size_t> polyface;
            size_t face_start = face_offsets[f];
            ufbx_face face = mesh->faces.data[f];
            for (size_t j = 0; j < face.num_indices; j++)
            {
                size_t corner_index = face_start + j;
                size_t vertex_index = corner_to_vertex[corner_index];
                polyface.push_back(start_pos + vertex_index);
            }
            face_indices.emplace_back(polyface);
        }
    }
    ufbx_free_scene(scene);
}
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *create_mesh(std::vector<MeshLib::HE_face<Real> *> component)
{
    if (component.empty())
        return 0;

    MeshLib::Mesh3D<Real> *m_pmesh = new MeshLib::Mesh3D<Real>;
    std::unordered_map<MeshLib::HE_vert<Real> *, MeshLib::HE_vert<Real> *> vert_map;
    for (auto f : component)
    {
        auto he = f->edge;
        do
        {
            if (vert_map.find(he->vert) == vert_map.end())
            {
                vert_map[he->vert] = m_pmesh->insert_vertex(he->vert->pos);
            }
            he = he->next;
        } while (he != f->edge);
    }

    for (auto f : component)
    {
        auto he = f->edge;
        std::vector<MeshLib::HE_vert<Real> *> face;
        do
        {
            face.push_back(vert_map[he->vert]);
            he = he->next;
        } while (he != f->edge);
        m_pmesh->insert_face(face);
    }

    m_pmesh->update_mesh();

    return m_pmesh;
}
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *create_mesh(const std::vector<TinyVector<Real, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, bool ignore_nonmanifold)
{
    if (!face_indices.empty())
    {
        MeshLib::Mesh3D<Real> *mesh = new MeshLib::Mesh3D<Real>();
        for (auto &v : vertices)
            mesh->insert_vertex(v);
        std::vector<MeshLib::HE_vert<Real> *> face;
        bool encountered_nonmanifold = false;
        for (auto &f : face_indices)
        {
            face.resize(0);
            for (auto &vid : f)
                face.emplace_back(mesh->get_vertex(vid));
            auto F = mesh->insert_face(face);
            if (!ignore_nonmanifold && !F)
            {
                encountered_nonmanifold = true;
                break;
            }
        }
        if (!ignore_nonmanifold && encountered_nonmanifold)
        {
            delete mesh;
            return 0;
        }
        mesh->update_mesh();
        return mesh;
    }
    return 0;
}
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *load_mesh(const std::string &filename, bool nonmanifold_input, bool ignore_nonmanifold)
{
    std::vector<TinyVector<Real, 3>> vertices;
    std::vector<std::vector<size_t>> face_indices;
    mesh_load_interface(filename, vertices, face_indices, nonmanifold_input);
    return create_mesh(vertices, face_indices, ignore_nonmanifold);
}
//////////////////////////////////////////////////////////////////////////
template <typename Real>
void mesh_decomposition(MeshLib::Mesh3D<Real> *mesh, std::vector<MeshLib::Mesh3D<Real> *> &submeshes)
{
    mesh->reset_faces_tag(false);
    for (int i = 0; i < mesh->get_num_of_faces(); i++)
    {
        auto face = mesh->get_face(i);
        if (face->tag)
            continue;

        std::queue<MeshLib::HE_face<Real> *> facequeue;
        facequeue.push(face);
        face->tag = true;
        std::vector<MeshLib::HE_face<Real> *> component;
        component.push_back(face);
        size_t num_non_tri = 0;
        while (!facequeue.empty())
        {
            auto f = facequeue.front();
            facequeue.pop();

            auto he = f->edge;
            do
            {
                if (he->pair->face && !he->pair->face->tag)
                {
                    facequeue.push(he->pair->face);
                    he->pair->face->tag = true;
                    component.push_back(he->pair->face);
                    if (he->pair->face->valence > 3)
                        num_non_tri++;
                }
                he = he->next;
            } while (he != f->edge);
        }

        MeshLib::Mesh3D<Real> *component_mesh = create_mesh(component);
        submeshes.push_back(component_mesh);
        // if (component_mesh->is_tri() || component_mesh->is_quad())
        //     submeshes.push_back(component_mesh);
        // else
        //     delete component_mesh;
    }
}
////////////////////////////////////////////////
template <typename Real>
MeshLib::Mesh3D<Real> *LoadPLYmesh(const std::string &filename, std::vector<std::array<unsigned char, 3>> &ply_face_color)
{
    MeshLib::Mesh3D<Real> *m_pmesh = 0;

    try
    {
        happly::PLYData plyIn(filename);

        std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
        std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();

        m_pmesh = new MeshLib::Mesh3D<Real>;
        for (size_t i = 0; i < vPos.size(); i++)
            m_pmesh->insert_vertex(TinyVector<Real, 3>(vPos[i][0], vPos[i][1], vPos[i][2]));
        typename MeshLib::Mesh3D<Real>::VERTEX_LIST vlist;
        bool encountered_nonmanifold = false;
        for (size_t i = 0; i < fInd.size(); i++)
        {
            vlist.resize(0);
            for (size_t j = 0; j < fInd[i].size(); j++)
                vlist.push_back(m_pmesh->get_vertex(fInd[i][j]));
            auto F = m_pmesh->insert_face(vlist);
            if (!F)
            {
                encountered_nonmanifold = true;
                break;
            }
        }
        if (encountered_nonmanifold)
        {
            delete m_pmesh;
            return 0;
        }
        m_pmesh->update_mesh();
        std::vector<std::array<double, 3>> vNormal = plyIn.getVertexNormals();
        std::vector<std::array<double, 3>> fNormal = plyIn.getVertexNormals("face");

        if (!vNormal.empty())
        {
            for (int i = 0; i < m_pmesh->get_num_of_vertices(); i++)
            {
                auto vert = m_pmesh->get_vertex(i);
                vert->normal[0] = vNormal[i][0], vert->normal[1] = vNormal[i][1], vert->normal[2] = vNormal[i][2];
            }
        }
        if (!fNormal.empty())
        {
            for (int i = 0; i < m_pmesh->get_num_of_faces(); i++)
            {
                auto face = m_pmesh->get_face(i);
                face->normal[0] = fNormal[i][0], face->normal[1] = fNormal[i][1], face->normal[2] = fNormal[i][2];
            }
        }

        if (plyIn.getElement("face").hasProperty("red") && plyIn.getElement("face").hasProperty("green") && plyIn.getElement("face").hasProperty("blue"))
        {
            std::vector<unsigned char> r = plyIn.getElement("face").getProperty<unsigned char>("red");
            std::vector<unsigned char> g = plyIn.getElement("face").getProperty<unsigned char>("green");
            std::vector<unsigned char> b = plyIn.getElement("face").getProperty<unsigned char>("blue");
            ply_face_color.resize(r.size());
            for (size_t i = 0; i < r.size(); i++)
            {
                ply_face_color[i][0] = r[i];
                ply_face_color[i][1] = g[i];
                ply_face_color[i][2] = b[i];
            }
        }

        return m_pmesh;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 0;
    }
} ////////////////////////////////////////////////
template <typename Real>
void mesh_to_vertices_and_faces(MeshLib::Mesh3D<Real> *m_pmesh, std::vector<TinyVector<Real, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices)
{
    vertices.resize(0), face_indices.resize(0);
    if (!m_pmesh)
        return;

    vertices.resize(m_pmesh->get_num_of_vertices());
    for (auto i = 0; i < m_pmesh->get_num_of_vertices(); i++)
    {
        auto v = m_pmesh->get_vertex(i);
        vertices[i] = v->pos;
    }

    face_indices.reserve(m_pmesh->get_num_of_faces());
    std::vector<size_t> face;

    for (auto i = 0; i < m_pmesh->get_num_of_faces(); i++)
    {
        auto f = m_pmesh->get_face(i);
        auto he = f->edge;
        face.resize(0);
        do
        {
            face.push_back((size_t)he->pair->vert->id);
            he = he->next;
        } while (he != f->edge);
        face_indices.emplace_back(face);
    }
}
////////////////////////////////////////////////
template void mesh_load_interface<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices, bool nonmanifold_input);
template void mesh_load_stl<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_ply<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_off<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_obj<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_glb<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_vtk<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_vtp<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_wrl<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);
template void mesh_load_fbx<double>(const std::string &filename, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);

template MeshLib::Mesh3D<double> *create_mesh(std::vector<MeshLib::HE_face<double> *> component);
template MeshLib::Mesh3D<double> *create_mesh(const std::vector<TinyVector<double, 3>> &vertices, const std::vector<std::vector<size_t>> &face_indices, bool ignore_nonmanifold);
template void mesh_decomposition(MeshLib::Mesh3D<double> *mesh, std::vector<MeshLib::Mesh3D<double> *> &submeshes);
template MeshLib::Mesh3D<double> *load_mesh(const std::string &filename, bool nonmanifold_input, bool ignore_nonmanifold);
template MeshLib::Mesh3D<double> *LoadPLYmesh(const std::string &filename, std::vector<std::array<unsigned char, 3>> &ply_face_color);
template void mesh_to_vertices_and_faces(MeshLib::Mesh3D<double> *m_pmesh, std::vector<TinyVector<double, 3>> &vertices, std::vector<std::vector<size_t>> &face_indices);