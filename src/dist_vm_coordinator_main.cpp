/**
 * dist_vm_coordinator_main.cpp
 *
 * Entry point for the VM-layer coordinator.
 *
 * Usage:
 *   dist-vm-coordinator [options]
 *
 *   --host <addr>        bind host (default 0.0.0.0)
 *   --control-port <n>   base coordinator control port (default 7700)
 *   --api-port <n>       inference API port (default 7702)
 *   --vm-port <n>        VM control port (default 7703)
 *   --model <path>       auto-assign model on startup
 *   --model-name <str>   model identifier sent to nodes
 *   --context <n>        context window (default 4096)
 *   --min-nodes <n>      wait for N nodes before auto-assign (default 1)
 */

#include "vm_coordinator.h"

#include <cstdlib>
#include <iostream>
#include <string>

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--host H] [--control-port P] "
              << "[--api-port P] [--vm-port P] [--model PATH] "
              << "[--model-name NAME] [--context N] [--min-nodes N]\n";
}

int main(int argc, char* argv[]) {
    dist::VmCoordinatorConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nxt = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if      (arg == "--host")         cfg.bind_host               = nxt();
        else if (arg == "--control-port") cfg.base.control_port       = (uint16_t)std::stoi(nxt());
        else if (arg == "--api-port")     cfg.base.api_port           = (uint16_t)std::stoi(nxt());
        else if (arg == "--vm-port")      cfg.vm_ctrl_port            = (uint16_t)std::stoi(nxt());
        else if (arg == "--model")        cfg.base.auto_model_path    = nxt();
        else if (arg == "--model-name")   cfg.base.auto_model_name    = nxt();
        else if (arg == "--context")      cfg.base.auto_n_ctx         = (uint32_t)std::stoi(nxt());
        else if (arg == "--min-nodes")    cfg.base.min_nodes          = (uint32_t)std::stoi(nxt());
        else { std::cerr << "Unknown flag: " << arg << "\n"; usage(argv[0]); return 1; }
    }

    cfg.base.bind_host = cfg.bind_host;

    std::cout << "[dist-vm-coordinator] starting\n"
              << "  control-port : " << cfg.base.control_port  << "\n"
              << "  api-port     : " << cfg.base.api_port      << "\n"
              << "  vm-port      : " << cfg.vm_ctrl_port       << "\n";

    dist::VmCoordinator coord(std::move(cfg));
    coord.run();
    return 0;
}
