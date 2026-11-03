cmake_minimum_required(VERSION 3.22)
project(reuse_resource_inventory LANGUAGES CXX)
add_library(reuse_resource_inventory SHARED resource_inventory.cpp)
if(CMAKE_HOST_WIN32)
  set(REUSE_SHADER_HOST windows-x86_64)
  set(REUSE_SHADER_EXE .exe)
elseif(CMAKE_HOST_APPLE)
  set(REUSE_SHADER_HOST darwin-x86_64)
else()
  set(REUSE_SHADER_HOST linux-x86_64)
endif()
set(REUSE_GLSLC "${ANDROID_NDK}/shader-tools/${REUSE_SHADER_HOST}/glslc${REUSE_SHADER_EXE}")
if(NOT EXISTS "${REUSE_GLSLC}")
  message(FATAL_ERROR "The pinned Android NDK shader compiler is missing: ${REUSE_GLSLC}")
endif()
set(REUSE_SHADER_INC "${CMAKE_CURRENT_BINARY_DIR}/generated/resource_smoke_spv.inc")
add_custom_command(OUTPUT "${REUSE_SHADER_INC}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
  COMMAND "${REUSE_GLSLC}" -fshader-stage=compute --target-env=vulkan1.0 -O -mfmt=c
          "${CMAKE_CURRENT_SOURCE_DIR}/resource_smoke.comp" -o "${REUSE_SHADER_INC}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resource_smoke.comp"
  VERBATIM)
target_sources(reuse_resource_inventory PRIVATE vulkan_probe.cpp "${REUSE_SHADER_INC}")
target_include_directories(reuse_resource_inventory PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_features(reuse_resource_inventory PRIVATE cxx_std_17)
target_compile_options(reuse_resource_inventory PRIVATE -Wall -Wextra -Werror)
target_link_libraries(reuse_resource_inventory PRIVATE dl)
