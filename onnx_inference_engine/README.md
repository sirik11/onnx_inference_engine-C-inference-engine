# ONNX Runtime Inference Engine

This directory contains a simple C++ inference engine built on top of **ONNX Runtime**.  The purpose is to provide a starting point for the **ONNX Runtime Inference Accelerator** project described in the planning document.  The engine loads an ONNX model, performs inference multiple times and exposes runtime metrics via Prometheus.

## Features

- **Modular C++ implementation:** uses the `onnxruntime_cxx_api` to load a model and run inference.
- **Quantisation flexibility:** you can supply FP32, FP16 or INT8 models.  ONNX Runtime supports dynamic quantisation for transformer models and static quantisation for CNNs; signed 8‑bit values with the QDQ format are recommended on CPU【565839220839977†L365-L396】.
- **Metrics instrumentation:** when built with [prometheus‑cpp](https://github.com/jupp0r/prometheus-cpp), the engine exports counters and histograms.  Metric naming follows Prometheus guidelines—use a domain prefix, base units (seconds) and suffix `_total` for counters【461519623353277†L100-L134】.
- **Profiling friendly:** compile with optimisations and profile using Linux `perf` (e.g. `perf stat` to collect CPU cycles and cache misses)【820774666345259†L65-L158】.  Thread affinity and memory alignment can be added to improve performance【83359744884238†L291-L342】.

## Prerequisites

- **ONNX Runtime** (v1.10+ recommended).  Download pre‑built binaries or build from source.  Set the environment variable `ONNXRUNTIME_DIR` to the root installation directory.
- **CMake** 3.16+ and a C++17 compiler (GCC/Clang/MSVC).
- (Optional) **prometheus‑cpp** library.  If found, metrics instrumentation will be enabled automatically.

## Building

```bash
# From this directory
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$ONNXRUNTIME_DIR"
make -j$(nproc)
```

The resulting executable will be `inference`.  If `prometheus-cpp` is installed and discoverable via `find_package`, metrics will be available at `http://localhost:8080/metrics`.

## Running

```bash
./inference path/to/model.onnx 5
```

This runs the model five times and prints the latency of each run.  If instrumentation is enabled, metrics will be exposed on port 8080.

## Quantisation tools

The `scripts/quantize.py` script demonstrates how to convert models to float16 or 8‑bit quantised versions using ONNX Runtime’s Python API.  Dynamic quantisation works well for RNN/transformer models; static quantisation with calibration data suits CNNs【565839220839977†L365-L396】.

## Thread affinity and alignment

To improve performance on multi‑socket machines, you can set custom thread affinities using the ONNX Runtime C++ API.  Pinning intra‑op threads to specific cores reduces NUMA crossing overhead and improved throughput by about 20 % in experiments【83359744884238†L291-L342】.  Further control can be achieved by calling `pthread_setaffinity_np` for each thread【119415834026756†L147-L229】.  Align dynamic arrays to 64‑byte boundaries and hint alignment to the compiler (`__assume_aligned`) to exploit AVX‑512 instructions【627265642849793†L105-L125】.

## Notes

- The sample code generates random input data.  Replace `generate_random_data` with actual input processing as needed.
- For INT8 models you may need to adjust input types and call `QuantizeStatic` or `QuantizeDynamic` functions in Python (`scripts/quantize.py`).
- This code provides a skeleton for the project; additional features such as batching, thread affinity management, advanced profiling and Prometheus metrics for memory usage can be added progressively as described in the plan.

