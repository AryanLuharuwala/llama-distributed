/**
 * dist_coordinator_main.cpp
 *
 * Usage:
 *   dist-coordinator [options]
 *
 *   -p, --control-port PORT     Node control port       (default: 7700)
 *   -a, --api-port PORT         Client inference port   (default: 7702)
 *   -d, --dashboard-port PORT   HTTP dashboard port     (default: 7780)
 *       --public-host HOST      Host shown in join command on dashboard
 *   -m, --model PATH            GGUF model to auto-load
 *   -n, --model-name NAME       Model name clients request by
 *   -c, --context N             Context window size     (default: 4096)
 *       --min-nodes N           Wait for N nodes before auto-assigning
 *   -h, --help
 */

#include "coordinator.h"
#include "platform_compat.h"

#include <iostream>
#include <string>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "  -p, --control-port PORT   Node control port      (default: 7700)\n"
        "  -a, --api-port PORT       Client API port        (default: 7702)\n"
        "  -d, --dashboard-port PORT HTTP dashboard port    (default: 7780, 0=off)\n"
        "      --public-host HOST    Host shown in join-URL on dashboard\n"
        "  -m, --model PATH          GGUF model to auto-load\n"
        "  -n, --model-name NAME     Model name\n"
        "  -c, --context N           Context size           (default: 4096)\n"
        "      --min-nodes N         Min nodes before auto-assigning\n"
        "  -h, --help\n",
        prog);
}

int main(int argc, char* argv[]) {
    dist::net_startup();
    dist::CoordinatorConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nxt = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };

        if      (arg == "-p" || arg == "--control-port")   cfg.control_port    = (uint16_t)std::stoi(nxt());
        else if (arg == "-a" || arg == "--api-port")        cfg.api_port        = (uint16_t)std::stoi(nxt());
        else if (arg == "-d" || arg == "--dashboard-port")  cfg.dashboard_port  = (uint16_t)std::stoi(nxt());
        else if (arg == "--public-host")                    cfg.public_host     = nxt();
        else if (arg == "-m" || arg == "--model")           cfg.auto_model_path = nxt();
        else if (arg == "-n" || arg == "--model-name")      cfg.auto_model_name = nxt();
        else if (arg == "-c" || arg == "--context")         cfg.auto_n_ctx      = (uint32_t)std::stoi(nxt());
        else if (arg == "--min-nodes")                      cfg.min_nodes       = (uint32_t)std::stoi(nxt());
        else if (arg == "--token-file")                     cfg.token_file      = nxt();
        else if (arg == "--region")                         cfg.region          = nxt();
        else if (arg == "--zone")                           cfg.zone            = nxt();
        else if (arg == "--issue-receipts")                 cfg.issue_receipts  = true;
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (!cfg.auto_model_path.empty() && cfg.auto_model_name.empty()) {
        std::string p = cfg.auto_model_path;
        size_t s = p.rfind('/'); if (s == std::string::npos) s = p.rfind('\\');
        std::string f = (s == std::string::npos) ? p : p.substr(s + 1);
        size_t d = f.rfind('.');
        cfg.auto_model_name = (d == std::string::npos) ? f : f.substr(0, d);
    }

    std::cout << "=== llama-distributed Coordinator ===\n"
              << "  control : " << cfg.control_port << "\n"
              << "  api     : " << cfg.api_port     << "\n"
              << "  dashboard: http://0.0.0.0:" << cfg.dashboard_port << "\n";
    if (!cfg.auto_model_path.empty())
        std::cout << "  model   : " << cfg.auto_model_path
                  << "  min-nodes: " << cfg.min_nodes << "\n";
    std::cout << "\n";

    try {
        dist::Coordinator coord(cfg);
        coord.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
