#include "model_converter.h"
#include <iostream>
#include <cstring>

static void print_usage(const char* prog) {
    std::cerr << "usage: " << prog << " -i <input_dir> -o <output.gguf> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "options:" << std::endl;
    std::cerr << "  -i, --input DIR      HuggingFace model directory" << std::endl;
    std::cerr << "  -o, --output FILE    Output GGUF file path" << std::endl;
    std::cerr << "  -t, --type TYPE      Output type: f32, f16 (default), q8_0, q4_1" << std::endl;
    std::cerr << "  -h, --help           Show this help" << std::endl;
}

int main(int argc, char** argv) {
    qwen3_asr::ConverterConfig cfg;
    cfg.output_type = "f16";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i < argc) cfg.input_dir = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i < argc) cfg.output_path = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--type") == 0) {
            if (++i < argc) cfg.output_type = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "error: unknown option: " << argv[i] << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.input_dir.empty() || cfg.output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    qwen3_asr::ModelConverter converter(cfg);
    if (!converter.convert()) {
        return 1;
    }

    return 0;
}
