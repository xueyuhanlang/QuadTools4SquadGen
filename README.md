# QuadTools

**Author:** Yang Liu (yangliu@microsoft.com)

## Description
**QuadTools** is a mesh processing toolkit developed for [our SQuadGen paper](https://youkang-kong.github.io/squadgen/) (SIGGRAPH 2026). It is used to generate the [training data](https://drive.google.com/file/d/1GthfgyHuYqhVeci7BJg-_6uc3QDH52IQ/view?usp=drive_link) for SQuadGen.

## Features

- **Triangle-to-Quad Conversion** — Merge triangle pairs into quads using a loop-aware matching algorithm
- **Quad Mesh Quality Evaluation** — Analyze quad layout complexity, loop simplicity, and other geometric quality metrics
- **CDF Data Generation** — Generate chart distance fields (CDF) from quad meshes for learning-based pipelines (see the SQuadGen paper)
- **Quad Extraction** — Extract quad meshes from CDF/DCDF parameterizations predicted by neural networks or derived from ground truth
- **Format Conversion** — Convert between mesh formats (OBJ, PLY, OFF, GLB, STL, FBX, VTK, WRL, and more)
- **Vertex Merging** — Merge duplicate vertices along mesh seams and resolve non-manifold edges caused by inconsistent welding, reducing topological defects and closing gaps

## Building

### Requirements

- CMake ≥ 3.12
- C++17 compatible compiler (GCC, Clang, or MSVC)
- (Optional) OpenMP for parallelism

### Compile

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

To enable OpenMP (not compatible with Python multiprocessing):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON
```

## Usage

All tools are command-line executables. Run any tool with `-h` to see the full list of options; **only the most common options are shown below**.

> `PLY` is recommended as the preferred I/O format due to its compact binary storage and flexible property handling.

> Although the code can load `GLB`/`FBX` files through third-party libraries, I recommend converting `GLB` to `PLY` with the Blender Python API for more robust handling of complex scenes.

### FormatConvert — Mesh Format Conversion

Convert meshes between different file formats.

```bash
./build/FormatConvert -i ./samples/duck_tri.ply -o ./samples/duck_tri.glb
```

- **Input formats:** PLY, OFF, OBJ, GLB/GLTF, STL, WRL, FBX
- **Output formats:** PLY, OFF, OBJ, GLB/GLTF, VTK, VTP, STL, USD, WRL, X3D

### Convert2Manifold — Handle Seams and Nonmanifold Issues via Vertex Merging

Convert non-manifold or seam-containing meshes to manifold. **Components that remain non-manifold after processing are removed from the output.** The tool merges vertices across boundaries to fix careless welding artifacts; it is not designed for robust non-manifold resolution via vertex/edge splitting.

```bash
./build/Convert2Manifold -i ./samples/aero.ply -o ./samples/aero_converted.ply
```
- **Input formats:** PLY, OFF, OBJ, GLB/GLTF, STL, WRL, FBX
- **Output formats:** PLY, OFF, OBJ, GLB/GLTF, VTK, VTP, STL, USD, WRL, X3D

### LoopQuad — Triangle-to-Quad Conversion

**LoopQuad** takes a polygonal mesh as input, converts triangle pairs into quads, and outputs only the components that form pure quad meshes with loop simplicity at or above the specified threshold. Duplicate quad components that are rigid-transformation copies of others are automatically removed. The output may be empty if no qualifying quad components are produced.

```bash
./build/LoopQuad -i ./samples/duck_tri.ply -o ./samples/duck_quad.ply -q 0.5
./build/LoopQuad -i ./samples/part_tri.ply -o ./samples/part_quad.ply 
```

| Option | Description                                             | Default |
| ------ | ------------------------------------------------------- | ------- |
| `-i`   | Input mesh (OBJ, OFF, PLY, GLB, STL, etc.)              | —       |
| `-o`   | Output quad mesh (OBJ, OFF, PLY)                        | —       |
| `-q`   | Loop simplicity threshold (0–1, higher = stricter)      | 0.8     |
| `-m`   | Minimum face count; skip components smaller than this   | 16      |
| `-s`   | Subdivide triangles into quads instead of merging       | false   |
| `-d`   | Debug mode (use to export non-pure-quad results) | false   |
| `-n`   | Enable non-manifold input handling                      | true    |


### QuadQuality — Mesh Quality Analysis

Evaluate quad mesh quality metrics including layout complexity, loop simplicity, and regularity.

```bash
./build/QuadQuality -i input.ply -j json_output_dir -d complex_dir -v
```
In the JSON file, `FratioN` and `EratioN` correspond to the face-loop simplicity and edge-loop simplicity scores, respectively.
Use `-v` to print the summary to the console.
You can visualize the colorized quad layouts by dragging files from `complex_dir` into [MeshLab](https://www.meshlab.net/).

| Option | Description                                                     | Default |
| ------ | --------------------------------------------------------------- | ------- |
| `-i`   | Input quad mesh                                                 | —       |
| `-o`   | Output directory (dump submeshes), optional                     | —       |
| `-j`   | JSON output folder for quality metrics, optional                | —       |
| `-d`   | Folder for dumping layout complexity data (for visualization), optional | -       |
| `-m`   | Dump layout complexity for the component with the most faces | true    |
| `-v`   | Verbose output                                                  | false   |


### CDFGen — CDF Training Data Generation

For each component of the input quad mesh, **CDFGen** generates chart distance field (CDF) data and stores it as an NPZ file for network training.

```bash
./build/CDFGen -i ./samples/duck_quad.ply -d ./samples/npz
```

| Option | Description                                                                      | Default |
| ------ | -------------------------------------------------------------------------------- | ------- |
| `-i`   | Input quad mesh (OBJ, OFF, PLY)                                                  | —       |
| `-d`   | Output directory for NPZ files                                                   | —       |
| `-f`   | Output filename prefix  (optional)                                               | —       |
| `-n`   | Number of sample points                                                          | 50000   |
| `-a`   | Sharp feature angle (degrees)                                                    | 130     |
| `-s`   | Random seed (int) for point sampling                                             | 0       |
| `-q`   | Write additional information into NPZ files for debugging the quad extraction code | false   |

The NPZ file can be visualized using `python/npzloader.py` which converts CDF/DCDF to textured OBJ files as well as GLB files:

```bash
# Install the required Python packages
pip install numpy point_cloud_utils trimesh matplotlib
# Export CDF as a textured mesh
python ./python/npzloader.py  ./samples/npz/duck_quad_0.npz --output-texture  --resolution 1024 --color-pattern 'cdf' --textured-obj-file-name ./samples/npz/vis/duck_quad_cdf.obj
# Export DCDF as a textured mesh
python ./python/npzloader.py  ./samples/npz/duck_quad_0.npz --output-texture  --resolution 1024 --color-pattern 'dcdf' --textured-obj-file-name ./samples/npz/vis/duck_quad_dcdf.obj
```
By specifying `--div 1` or `--div 2`, you can export the densified CDF/DCDF as shown in the paper appendix.



### QuadExtraction — Extract Quads from Parameterization

Extract quad meshes from CDF/DCDF parameterizations.

```bash
./build/QuadExtraction -i subdivided.ply -f features.npz -o outputquad.ply --ringsize=8 --verbose=false -a 150 --div=0 -t 0.1 -s 3 -r 15
```
`subdivided.ply` is the densely subdivided version of the original input triangle mesh, and `features.npz` stores the CDF/DCDF values at its triangle faces and/or vertices along with offset information.

 - You can simulate these two files by specifying the `-q` option when running CDFGen; a file named `subdiv.ply` will be generated alongside the NPZ file in the specified output folder.
  Example:
  ```bash 
      ./build/CDFGen -i ./samples/duck_quad.ply -d ./samples/npz -q

      ./build/QuadExtraction -i ./samples/npz/subdiv.ply -f ./samples/npz/duck_quad_0.npz -o ./samples/npz/quadoutput.ply --ringsize=8 --verbose=false -a 150 --div=0 -t 0.1 -s 3 -r 15
  ```

- For the SQuadGen neural network, the Python code creates `subdivided.ply` and `features.npz` automatically.


| Option       | Description                                          | Default |
| ------------ | ---------------------------------------------------- | ------- |
| `-i`         | Input triangle mesh                                  | —       |
| `-f`         | Feature input (NPZ)                                  | —       |
| `-o`         | Output quad mesh                                     | —       |
| `--div`      | Color pattern division (int); use 0 or 1 in practice | 0       |
| `-t`         | Color binarization threshold                         | 0.1     |
| `-s`         | Number of subdivisions                               | 0       |
| `-r`         | Number of smoothing iterations                       | 0       |
| `-a`         | Feature angle threshold                              | 150     |
| `--ringsize` | Ring size for color enhancement                      | 10      |
| `--verbose`  | Print log                                            | true    |

## Project Structure

```
QuadTools/
├── CMakeLists.txt          # Build configuration
├── src/
│   ├── meshlib/            # Half-edge mesh data structure, I/O, spatial queries, and various operations
│   ├── quadlib/            # Algorithm modules (loopquad, cdfgen, quadextraction, ...)
│   └── external/           # Vendored dependencies (OpenMesh, blossom5, nanoflann, etc.)
├── driver/                 # CLI tool entry points (*_main.cpp)
├── python/                 # Python code for data processing and visualization
├── samples/                # Sample data
└── build/                  # Build output

```
## Reference

If you use this code in your research, please cite:

```bibtex
@article{kong2026squadgen,
  title= {SQuadGen: Generating Simple Quad Layouts via Chart Distance Fields},
  author= {Kong, Youkang and Liu, Yang and Dong, Yue and Tong, Xin and Shum, Heung-Yeung},
  journal= {ACM Transactions on Graphics (SIGGRAPH)},
  volume = {45},
  number = {4},
  pages = {144:1-144:15},
  year={2026}
}
```

### Third Party Code
This project vendors multiple third-party libraries under `src/external`.

For the full dependency list (including versions and usage), see [`src/external/readme.md`](src/external/readme.md).

## License

This project is licensed under the [MIT License](LICENSE).