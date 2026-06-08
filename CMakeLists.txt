cmake_minimum_required(VERSION 3.12)
project(QUADTOOLS LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# define using Openmp or not
option(USE_OPENMP "Use OpenMP" OFF)

# set USE_OPENMP to OFF if you want to call from Python with multiprocessing. It seems that OpenMP in C++ has conflict with Python multiprocessing.

# Set OpenMP libraries conditionally
set(OPENMP_LIBS "")

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(OPENMP_LIBS OpenMP::OpenMP_CXX)
else()
    find_package(Threads REQUIRED)
    set(OPENMP_LIBS Threads::Threads)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# ####################################################
FILE(GLOB_RECURSE project_cpps src/*.cpp src/*.c src/external/OpenMesh/*.cc)

INCLUDE_DIRECTORIES(src/meshlib src/quadlib src/quadlib/loopquad src/quadlib/loopquality src/quadlib/blockgen src/quadlib/cdfgen src/quadlib/quadextraction  src/quadlib/quadimprove src/external src/external/coloring src/external/blossom5 src/external/ANN src/external/tinygltf src/external/rapidjson src/external/Libboard src/external/ufbx src/external/OpenMesh)

# ####################################################
# if using MSVC, set static library
if(MSVC)
    ADD_LIBRARY(QuadTools STATIC ${project_cpps})
else(MSVC)
    ADD_LIBRARY(QuadTools SHARED ${project_cpps})
endif(MSVC)

set_target_properties(QuadTools PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC) # required by OpenMesh
    target_compile_options(QuadTools PRIVATE /bigobj)
    target_compile_definitions(QuadTools PUBLIC _USE_MATH_DEFINES)
endif()

target_link_libraries(QuadTools PRIVATE ${OPENMP_LIBS})

# ####################################################
ADD_EXECUTABLE(LoopQuad driver/loopquad_main.cpp)
target_link_libraries(LoopQuad PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(Unique driver/unique_main.cpp)
target_compile_definitions(Unique PRIVATE UNIQUE)
target_link_libraries(Unique PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(Unique_info driver/unique_main.cpp)
target_compile_definitions(Unique_info PRIVATE WRITE_INFO)
target_link_libraries(Unique_info PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(Unique_writeply driver/unique_main.cpp)
target_compile_definitions(Unique_writeply PRIVATE WRITE_PLY)
target_link_libraries(Unique_writeply PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(Convert2Manifold driver/convert2manifold_main.cpp)
target_link_libraries(Convert2Manifold PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(BlockGen driver/blockgen_main.cpp)
target_link_libraries(BlockGen PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(Ply2SVG driver/ply2svg_main.cpp)
target_link_libraries(Ply2SVG PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(CDFGen driver/cdfgen_main.cpp)
target_link_libraries(CDFGen PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(QuadQuality driver/quadquality_main.cpp)
target_link_libraries(QuadQuality PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(QuadExtraction driver/quadextraction_main.cpp)
target_link_libraries(QuadExtraction PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(SplitMesh driver/splitmesh_main.cpp)
target_link_libraries(SplitMesh PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(LabelComponent driver/labelcomponent_main.cpp)
target_link_libraries(LabelComponent PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(FormatConvert driver/formatconverter_main.cpp)
target_link_libraries(FormatConvert PRIVATE ${OPENMP_LIBS} QuadTools)

# ####################################################
ADD_EXECUTABLE(QuadImprove driver/quadimprove_main.cpp)
target_link_libraries(QuadImprove PRIVATE ${OPENMP_LIBS} QuadTools)