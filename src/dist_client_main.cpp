/**
 * dist_client_main.cpp
 *
 * CLI client that sends inference requests to the Coordinator API port
 * and streams generated tokens to stdout.
 *
 * Usage:
 *   dist-client -s COORDINATOR_HOST -m MODEL_NAME -p "Your prompt here"
 *
 *   -s, --server HOST         Coordinator host (required)
 *   -a, --api-port PORT       API port (default: 7702)
 *   -m, --model NAME          Model name to use
 *   -p, --prompt TEXT         Prompt text (or use stdin if not given)
 *       --max-tokens N        Max tokens to generate (default: 256)
 *       --temp F              Temperature (default: 0.7)
 *       --top-p F             Top-p (default: 0.9)
 *   -h, --help
 *
 * Token ids are computed by a simple whitespace tokenizer for illustration.
 * In production, use llama_tokenize() locally.
 */

#include "dist_protocol.h"
#include "dist_conn.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s -s HOST -m MODEL -p PROMPT [options]\n"
        "\n"
        "  -s, --server HOST         Coordinator host (required)\n"
        "  -a, --api-port PORT       API port (default: 7702)\n"
        "  -m, --model NAME          Model name\n"
        "  -p, --prompt TEXT         Prompt text\n"
        "      --max-tokens N        Max tokens to generate (default: 256)\n"
        "      --temp F              Temperature (default: 0.7)\n"
        "      --top-p F             Top-p (default: 0.9)\n"
        "  -h, --help\n",
        prog);
}

// Trivial byte-level tokenizer for demo purposes.
// In a real deployment, link llama.cpp and call llama_tokenize().
static std::vector<int32_t> simple_tokenize(const std::string& text) {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back((int32_t)c);
    return ids;
}

int main(int argc, char* argv[]) {
    std::string  server_host;
    uint16_t     api_port     = dist::PORT_API;
    std::string  model_name;
    std::string  prompt;
    uint32_t     max_tokens   = 256;
    float        temperature  = 0.7f;
    float        top_p        = 0.9f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); exit(1); }
            return argv[i];
        };

        if      (arg == "-s" || arg == "--server")     server_host = next();
        else if (arg == "-a" || arg == "--api-port")   api_port    = (uint16_t)std::stoi(next());
        else if (arg == "-m" || arg == "--model")      model_name  = next();
        else if (arg == "-p" || arg == "--prompt")     prompt      = next();
        else if (arg == "--max-tokens")                max_tokens  = (uint32_t)std::stoi(next());
        else if (arg == "--temp")                      temperature = std::stof(next());
        else if (arg == "--top-p")                     top_p       = std::stof(next());
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (server_host.empty()) {
        fprintf(stderr, "Error: --server required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (prompt.empty()) {
        // Read from stdin
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        prompt = ss.str();
    }

    if (prompt.empty()) {
        fprintf(stderr, "Error: prompt is empty\n");
        return 1;
    }

    // Tokenize prompt
    std::vector<int32_t> token_ids = simple_tokenize(prompt);
    std::cout << "[client] prompt tokens: " << token_ids.size() << "\n";

    // Build INFER_REQUEST payload
    std::vector<uint8_t> payload(sizeof(dist::MsgInferRequest)
                                  + token_ids.size() * sizeof(int32_t));
    auto& req = *reinterpret_cast<dist::MsgInferRequest*>(payload.data());
    req.request_id      = 1;
    req.n_prompt_tokens = (uint32_t)token_ids.size();
    req.max_gen_tokens  = max_tokens;
    req.temperature     = temperature;
    req.top_p           = top_p;
    req.seed            = -1;
    strncpy(req.model_name, model_name.c_str(), sizeof(req.model_name) - 1);
    memcpy(payload.data() + sizeof(dist::MsgInferRequest),
           token_ids.data(),
           token_ids.size() * sizeof(int32_t));

    // Connect and send
    dist::Connection conn;
    try {
        conn.connect(server_host, api_port);
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect to " << server_host << ":" << api_port
                  << " — " << e.what() << "\n";
        return 1;
    }

    conn.send_msg(dist::MsgType::INFER_REQUEST, payload.data(), (uint32_t)payload.size());
    std::cout << "[client] request sent, waiting for tokens...\n\n";

    // Stream response
    dist::MsgHeader hdr{};
    std::vector<uint8_t> resp;

    auto t_start = std::chrono::steady_clock::now();
    bool first_token = true;
    double ttft_ms = 0.0;

    while (conn.recv_msg(hdr, resp)) {
        switch (static_cast<dist::MsgType>(hdr.msg_type)) {
        case dist::MsgType::INFER_TOKEN: {
            if (resp.size() < sizeof(dist::MsgInferToken)) break;
            const auto& tok = *reinterpret_cast<const dist::MsgInferToken*>(resp.data());
            if (first_token) {
                auto now = std::chrono::steady_clock::now();
                ttft_ms = std::chrono::duration<double, std::milli>(now - t_start).count();
                first_token = false;
            }
            // Print token as a character (byte tokenizer demo)
            if (tok.token_id < 128) {
                char c = (char)tok.token_id;
                std::cout << c << std::flush;
            } else {
                std::cout << "[" << tok.token_id << "]" << std::flush;
            }
            break;
        }
        case dist::MsgType::INFER_DONE: {
            if (resp.size() < sizeof(dist::MsgInferDone)) break;
            const auto& done = *reinterpret_cast<const dist::MsgInferDone*>(resp.data());
            std::cout << "\n\n[client] done."
                      << " tokens=" << done.n_tokens_generated
                      << " ttft=" << ttft_ms << "ms"
                      << " tps=" << done.tokens_per_second << "\n";
            return 0;
        }
        case dist::MsgType::INFER_ERROR: {
            if (resp.size() < sizeof(dist::MsgInferError)) break;
            const auto& err = *reinterpret_cast<const dist::MsgInferError*>(resp.data());
            std::cerr << "\n[client] ERROR: " << err.message << "\n";
            return 1;
        }
        default:
            break;
        }
    }

    std::cerr << "[client] connection closed\n";
    return 1;
}
