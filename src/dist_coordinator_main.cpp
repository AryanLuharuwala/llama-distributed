/**
 * dist_coordinator_main.cpp
 *
 * Entry point for the Pool Coordinator daemon.
 *
 * Usage:
 *   dist-coordinator [options]
 *
 *   -p, --control-port PORT     Control plane port (default: 7700)
 *   -a, --api-port PORT         Client API port (default: 7702)
 *   -m, --model PATH            Auto-load this model on startup
 *   -n, --model-name NAME       Model name (used by clients to request)
 *   -c, --context N             Context window size (default: 4096)
 *   --min-nodes N               Wait for N nodes before auto-assigning (default: 1)
 *   -h, --help
 */

#include "coordinator.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "  -p, --control-port PORT   Node control port (default: 7700)\n"
        "  -a, --api-port PORT       Client API port   (default: 7702)\n"
        "  -m, --model PATH          GGUF model to auto-load\n"
        "  -n, --model-name NAME     Model name clients use to request\n"
        "  -c, --context N           Context size (default: 4096)\n"
        "      --min-nodes N         Min nodes before auto-assigning (default: 1)\n"
        "  -h, --help\n",
        prog);
}

int main(int argc, char* argv[]) {
    dist::CoordinatorConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };

        if (arg == "-p" || arg == "--control-port") {
            cfg.control_port = (uint16_t)std::stoi(next());
        } else if (arg == "-a" || arg == "--api-port") {
            cfg.api_port = (uint16_t)std::stoi(next());
        } else if (arg == "-m" || arg == "--model") {
            cfg.auto_model_path = next();
        } else if (arg == "-n" || arg == "--model-name") {
            cfg.auto_model_name = next();
        } else if (arg == "-c" || arg == "--context") {
            cfg.auto_n_ctx = (uint32_t)std::stoi(next());
        } else if (arg == "--min-nodes") {
            cfg.min_nodes = (uint32_t)std::stoi(next());
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!cfg.auto_model_path.empty() && cfg.auto_model_name.empty()) {
        // Default model name = filename without extension
        std::string p = cfg.auto_model_path;
        size_t slash = p.rfind('/');
        if (slash == std::string::npos) slash = p.rfind('\\');
        std::string fname = (slash == std::string::npos) ? p : p.substr(slash + 1);
        size_t dot = fname.rfind('.');
        cfg.auto_model_name = (dot == std::string::npos) ? fname : fname.substr(0, dot);
    }

    std::cout << "=== llama-distributed Coordinator ===\n";
    std::cout << "control_port=" << cfg.control_port
              << " api_port=" << cfg.api_port << "\n";
    if (!cfg.auto_model_path.empty())
        std::cout << "auto_model=" << cfg.auto_model_path
                  << " min_nodes=" << cfg.min_nodes << "\n";

    try {
        dist::Coordinator coord(cfg);
        coord.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
