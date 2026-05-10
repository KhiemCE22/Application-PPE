#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <set>

/**
 * Simple NCNN Param Parser (Zero Dependencies)
 * This tool reads the text-based .param file to find inputs and outputs.
 */

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [model.param]" << std::endl;
        return -1;
    }

    std::ifstream fs(argv[1]);
    if (!fs.is_open()) {
        std::cerr << "Error: Could not open file " << argv[1] << std::endl;
        return -1;
    }

    std::string line;
    // Skip the first two header lines (Magic number and Layer/Blob counts)
    std::getline(fs, line); 
    std::getline(fs, line);

    std::vector<std::string> all_outputs;
    std::set<std::string> consumed_inputs;
    std::vector<std::string> model_inputs;

    while (std::getline(fs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string layer_type, layer_name;
        int input_count, output_count;

        ss >> layer_type >> layer_name >> input_count >> output_count;

        // If it's an Input layer, record it
        if (layer_type == "Input") {
            model_inputs.push_back(layer_name);
        }

        // Track inputs consumed by this layer
        for (int i = 0; i < input_count; i++) {
            std::string blob_name;
            ss >> blob_name;
            consumed_inputs.insert(blob_name);
        }

        // Track outputs produced by this layer
        for (int i = 0; i < output_count; i++) {
            std::string blob_name;
            ss >> blob_name;
            all_outputs.push_back(blob_name);
        }
    }

    std::cout << "--- NCNN Model Structure (Text Parse) ---" << std::endl;
    
    std::cout << "[INPUTS]:" << std::endl;
    for (const auto& in : model_inputs) {
        std::cout << "  -> " << in << std::endl;
    }

    std::cout << "[OUTPUTS (Final Blobs)]:" << std::endl;
    for (const auto& out : all_outputs) {
        // If an output is never consumed by another layer, it is a model exit point
        if (consumed_inputs.find(out) == consumed_inputs.end()) {
            std::cout << "  <- " << out << std::endl;
        }
    }

    return 0;
}