cmake_minimum_required(VERSION 3.27)
project(maple_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(MLX CONFIG REQUIRED)
# MLX wheels can export SDK paths from the wheel builder's machine. Resolve
# the same frameworks using the local SDK instead of those absolute paths.
foreach(framework Metal Foundation QuartzCore Accelerate)
  find_library(MAPLE_${framework}_FRAMEWORK ${framework} REQUIRED)
  list(APPEND MAPLE_FRAMEWORKS "${MAPLE_${framework}_FRAMEWORK}")
endforeach()
set_property(TARGET mlx PROPERTY INTERFACE_LINK_LIBRARIES "${MAPLE_FRAMEWORKS}")
# Match the per-CPython nanobind ABI used by MLX wheels.
nanobind_add_module(_maple_native NB_STATIC LTO NOMINSIZE NB_DOMAIN mlx native.cpp)
target_link_libraries(_maple_native PRIVATE mlx)
set_target_properties(_maple_native PROPERTIES
  BUILD_RPATH "${MLX_LIBRARY_DIRS}"
  INSTALL_RPATH "@loader_path/../mlx/lib")
