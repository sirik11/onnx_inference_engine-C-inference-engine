# onnx_inference_engine-C-inference-engine

Purpose: This folder contains a minimal but extensible C++ application that loads an ONNX model and runs inference using ONNX Runtime. It demonstrates how to support multiple precisions (FP32, FP16, INT8), how to expose Prometheus metrics and provides a Python helper to convert models.
Key files and their roles:
src/main.cpp – The core program. It:
Initialises an ONNX Runtime session and loads the model.
Generates random input data and runs inference in a loop.
Records the latency of each run and, when built with the Prometheus client library, exposes counters and histograms (inference_total and inference_latency_seconds) on port 8080. Metric names follow Prometheus conventions for units and suffixes
prometheus.io
.
Can be extended to implement batching, thread affinity and memory alignment; ONNX Runtime docs show that pinning threads to NUMA nodes can improve performance by ~20 %
onnxruntime.ai
 and that aligning data to 64‑byte boundaries benefits SIMD code
intel.com
.
CMakeLists.txt – Build script that:
Sets the project to use C++17.
Looks up ONNX Runtime and Prometheus‑cpp headers/libraries. If prometheus-cpp is found, a macro enables metrics instrumentation.
Compiles src/main.cpp into an executable called inference.
scripts/quantize.py – A Python script to convert your FP32 model to FP16 or INT8. It uses ONNX Runtime’s quantisation API:
--type fp16 calls convert_float_to_float16
onnxruntime.ai
.
--type dynamic applies dynamic 8‑bit quantisation, which is recommended for RNN/transformer models
onnxruntime.ai
.
--type static performs static quantisation with a dummy calibration reader (you should replace this with real calibration data for CNNs). Static quantisation is recommended for CNNs
onnxruntime.ai
.
README.md – Step‑by‑step instructions:
Lists prerequisites (ONNX Runtime, CMake, optional prometheus‑cpp).
Shows how to run cmake .. and make to build the engine.
Explains how to run the compiled executable, convert models with quantize.py, and why thread affinity and memory alignment matter
onnxruntime.ai
intel.com
.
Getting started:
Install ONNX Runtime and CMake; on macOS, use the Xcode command‑line tools and Homebrew (brew install cmake).
Extract or clone the repo and run:
cd projects/onnx_inference_engine
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$ONNXRUNTIME_DIR"
make
Use scripts/quantize.py to produce FP16 or INT8 models, then run:
./inference path/to/model_fp16.onnx 10
If you built with Prometheus support, visit http://localhost:8080/metrics to see runtime counters and histograms.
