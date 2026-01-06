#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>

#ifdef WITH_PROMETHEUS
#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/counter.h>
#include <prometheus/histogram.h>
#endif

// Simple utility to generate dummy input tensor data
std::vector<float> generate_random_data(size_t size) {
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(size);
    for(auto &v : data) {
        v = dist(gen);
    }
    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [num_runs]\n";
        return 1;
    }
    const char* model_path = argv[1];
    size_t num_runs = (argc > 2) ? std::stoul(argv[2]) : 10;

#ifdef WITH_PROMETHEUS
    // Set up an exposer to expose metrics on port 8080
    prometheus::Exposer exposer{"0.0.0.0:8080"};
    auto registry = std::make_shared<prometheus::Registry>();

    auto& inference_family = prometheus::BuildCounter()
        .Name("inference_total")
        .Help("Number of inferences processed")
        .Register(*registry);
    auto& inference_counter = inference_family.Add({});

    auto& latency_family = prometheus::BuildHistogram()
        .Name("inference_latency_seconds")
        .Help("Inference latency in seconds")
        .Register(*registry);
    // buckets in seconds
    std::vector<double> buckets{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0};
    auto& latency_histogram = latency_family.Add({}, buckets);

    // Expose metrics
    exposer.RegisterCollectable(registry);
#endif

    // Initialize ONNX Runtime environment and session options
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "inference"};
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the model
    Ort::Session session{env, model_path, session_options};
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input node information
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<std::vector<int64_t>> input_node_dims(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        char* input_name = session.GetInputName(i, allocator);
        input_node_names[i] = input_name;
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims[i] = tensor_info.GetShape();
    }

    // Generate dummy data for the first input (assume single input)
    size_t input_size = 1;
    for (auto dim : input_node_dims[0]) {
        input_size *= (dim > 0) ? dim : 1;
    }
    std::vector<float> input_data = generate_random_data(input_size);

    std::vector<int64_t> input_shape = input_node_dims[0];

    // Prepare input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // Prepare output names
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_names(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }

    // Run inference multiple times
    for (size_t i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_node_names.data(), &input_tensor, 1,
                                          output_node_names.data(), num_output_nodes);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

#ifdef WITH_PROMETHEUS
        inference_counter.Increment();
        latency_histogram.Observe(elapsed.count());
#endif

        std::cout << "Run " << i << ": inference completed in " << elapsed.count() << " seconds." << std::endl;
    }

    std::cout << "Finished " << num_runs << " runs." << std::endl;
    return 0;
}
