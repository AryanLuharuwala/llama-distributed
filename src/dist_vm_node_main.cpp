/**
 * dist_vm_node_main.cpp
 *
 * Entry point for a VM-layer worker node.
 *
 * Usage:
 *   dist-vm-node [options]
 *
 *   --server <host>       VmCoordinator hostname / IP (required)
 *   --control-port <n>    coordinator control port (default 7700)
 *   --data-port <n>       local pipeline data port (default 7701)
 *   --vm-port <n>         coordinator VM ctrl port (default 7703)
 *   --n-gpu-layers <n>    layers to offload to GPU (default 999 = all)
 *   --context <n>         context window size (default 4096)
 *   --batch <n>           max batch size (default 512)
 *   --id <str>            node identifier (default hostname:pid)
 */

#include "vm_node.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>   // gethostname

static std::string default_node_id() {
    char host[256] = {};
    gethostname(host, sizeof(host));
    return std::string(host) + ":" + std::to_string(getpid());
}

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --server HOST [--control-port P] [--data-port P]"
                 " [--vm-port P] [--n-gpu-layers N] [--context N]"
                 " [--batch N] [--id ID]\n";
}

int main(int argc, char* argv[]) {
    dist::VmNodeConfig cfg;
    cfg.base.node_id = default_node_id();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nxt = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if      (arg == "--server")       { cfg.base.coordinator_host = nxt();
                                            cfg.vm_coordinator_host   = cfg.base.coordinator_host; }
        else if (arg == "--control-port") cfg.base.coordinator_port   = (uint16_t)std::stoi(nxt());
        else if (arg == "--data-port")    cfg.base.data_port          = (uint16_t)std::stoi(nxt());
        else if (arg == "--vm-port")      cfg.vm_ctrl_port            = (uint16_t)std::stoi(nxt());
        else if (arg == "--n-gpu-layers") cfg.base.n_gpu_layers       = std::stoi(nxt());
        else if (arg == "--context")      cfg.base.n_ctx              = (uint32_t)std::stoi(nxt());
        else if (arg == "--batch")        cfg.base.n_batch            = (uint32_t)std::stoi(nxt());
        else if (arg == "--id")           { cfg.base.node_id          = nxt();  }
        else { std::cerr << "Unknown flag: " << arg << "\n"; usage(argv[0]); return 1; }
    }

    if (cfg.base.coordinator_host.empty()) {
        std::cerr << "Error: --server is required\n";
        usage(argv[0]);
        return 1;
    }

    std::cout << "[dist-vm-node] id=" << cfg.base.node_id
              << " server=" << cfg.base.coordinator_host
              << " data-port=" << cfg.base.data_port
              << " vm-port=" << cfg.vm_ctrl_port << "\n";

    dist::VmNode node(std::move(cfg));
    node.run();
    return 0;
}
