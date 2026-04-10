/**
 * dist_node_main.cpp
 *
 * Entry point for a Node Agent — one per GPU server.
 *
 * Usage:
 *   dist-node -s COORDINATOR_HOST [options]
 *
 *   -s, --server HOST           Coordinator host (required)
 *   -p, --control-port PORT     Coordinator control port (default: 7700)
 *   -d, --data-port PORT        This node's data port (default: 7701)
 *   -g, --n-gpu-layers N        GPU layers (default: 999 = all)
 *   -c, --context N             Context window (default: 4096)
 *   -b, --batch N               Batch size (default: 512)
 *   --id NAME                   Node ID override (default: hostname:pid)
 *   -h, --help
 */

#include "node_agent.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s -s COORDINATOR_HOST [options]\n"
        "\n"
        "  -s, --server HOST         Coordinator host (required)\n"
        "  -p, --control-port PORT   Coordinator control port (default: 7700)\n"
        "  -d, --data-port PORT      This node's data listen port (default: 7701)\n"
        "  -g, --n-gpu-layers N      GPU layers to offload (default: 999=all)\n"
        "  -c, --context N           Context window size (default: 4096)\n"
        "  -b, --batch N             Batch size (default: 512)\n"
        "      --id NAME             Override node ID\n"
        "  -h, --help\n",
        prog);
}

int main(int argc, char* argv[]) {
    dist::NodeAgentConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };

        if (arg == "-s" || arg == "--server") {
            cfg.coordinator_host = next();
        } else if (arg == "-p" || arg == "--control-port") {
            cfg.coordinator_port = (uint16_t)std::stoi(next());
        } else if (arg == "-d" || arg == "--data-port") {
            cfg.data_port = (uint16_t)std::stoi(next());
        } else if (arg == "-g" || arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = std::stoi(next());
        } else if (arg == "-c" || arg == "--context") {
            cfg.n_ctx = (uint32_t)std::stoi(next());
        } else if (arg == "-b" || arg == "--batch") {
            cfg.n_batch = (uint32_t)std::stoi(next());
        } else if (arg == "--id") {
            cfg.node_id = next();
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.coordinator_host.empty()) {
        fprintf(stderr, "Error: --server is required\n");
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== llama-distributed Node Agent ===\n";
    std::cout << "coordinator=" << cfg.coordinator_host << ":"
              << cfg.coordinator_port
              << " data_port=" << cfg.data_port << "\n";

    try {
        dist::NodeAgent agent(cfg);
        agent.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
