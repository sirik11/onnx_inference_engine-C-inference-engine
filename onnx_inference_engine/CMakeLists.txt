cmake_minimum_required(VERSION 3.16)
project(onnx_inference_engine LANGUAGES CXX)

# Set C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime.  Users should set ONNXRUNTIME_DIR or install it system‑wide.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIB onnxruntime)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "ONNX Runtime headers/libraries not found. Set ONNXRUNTIME_DIR or install ONNX Runtime.")
endif()

# Find Prometheus‑cpp
find_path(PROMETHEUS_CPP_INCLUDE_DIR prometheus/exposer.h)
find_library(PROMETHEUS_CPP_CORE_LIB prometheus-cpp-core)
find_library(PROMETHEUS_CPP_PULL_LIB prometheus-cpp-pull)

# If Prometheus‑cpp is unavailable, build will still succeed but instrumentation will be disabled.
if(PROMETHEUS_CPP_INCLUDE_DIR AND PROMETHEUS_CPP_CORE_LIB AND PROMETHEUS_CPP_PULL_LIB)
  add_definitions(-DWITH_PROMETHEUS)
  message(STATUS "Found Prometheus‑cpp: enabling metrics instrumentation")
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
if(PROMETHEUS_CPP_INCLUDE_DIR)
  include_directories(${PROMETHEUS_CPP_INCLUDE_DIR})
endif()

add_executable(inference
  src/main.cpp
)

target_link_libraries(inference PRIVATE ${ONNXRUNTIME_LIB})
if(PROMETHEUS_CPP_CORE_LIB AND PROMETHEUS_CPP_PULL_LIB)
  target_link_libraries(inference PRIVATE ${PROMETHEUS_CPP_CORE_LIB} ${PROMETHEUS_CPP_PULL_LIB})
endif()
