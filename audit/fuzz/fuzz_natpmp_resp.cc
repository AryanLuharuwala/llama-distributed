// Fuzz the response-parsing branches of src/nat_pmp.cpp.
//
// We can't fuzz the network round-trip easily, so this harness lifts the
// response-parsing logic out of try_pcp_map() and try_natpmp_map() into a
// pure function and runs the fuzzer over the response bytes.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <arpa/inet.h>

namespace {

// Mirror of the response handler in try_pcp_map().  Input: response bytes.
// Returns whether parsing produced a valid mapping (true) or rejected (false).
bool parse_pcp_response(const uint8_t* resp, ssize_t k) {
    if (k < 60) return false;
    if (resp[0] != 2) return false;
    if ((resp[1] & 0x7F) != 1) return false;
    uint8_t result = resp[3];
    if (result != 0) return false;
    uint32_t granted = ((uint32_t)resp[8] << 24) | ((uint32_t)resp[9] << 16) |
                       ((uint32_t)resp[10] << 8) |  (uint32_t)resp[11];
    (void)granted;
    uint16_t ext_port = ((uint16_t)resp[42] << 8) | resp[43];
    (void)ext_port;

    bool v4mapped = true;
    for (int i = 0; i < 10; ++i) if (resp[44+i] != 0) { v4mapped = false; break; }
    if (resp[54] != 0xff || resp[55] != 0xff) v4mapped = false;

    char buf[INET6_ADDRSTRLEN];
    if (v4mapped) {
        struct in_addr a;
        std::memcpy(&a.s_addr, &resp[56], 4);
        inet_ntop(AF_INET, &a, buf, sizeof(buf));
    } else {
        struct in6_addr a6;
        std::memcpy(&a6, &resp[44], 16);
        inet_ntop(AF_INET6, &a6, buf, sizeof(buf));
    }
    return true;
}

// Mirror of try_natpmp_map response handler.
bool parse_natpmp_response(const uint8_t* resp, ssize_t k) {
    if (k < 16) return false;
    if (resp[0] != 0) return false;
    if (resp[1] != (128 + 1)) return false;
    uint16_t result = ((uint16_t)resp[2] << 8) | resp[3];
    if (result != 0) return false;
    uint16_t int_port = ((uint16_t)resp[8]  << 8) | resp[9];
    uint16_t ext_port = ((uint16_t)resp[10] << 8) | resp[11];
    uint32_t granted  = ((uint32_t)resp[12] << 24) | ((uint32_t)resp[13] << 16) |
                        ((uint32_t)resp[14] << 8)  |  (uint32_t)resp[15];
    (void)int_port; (void)ext_port; (void)granted;
    return true;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Both response buffers are uint8_t[64] / uint8_t[1100] in the real code;
    // the parsers only ever read up to fixed offsets gated by length checks.
    parse_pcp_response(data, (ssize_t)size);
    parse_natpmp_response(data, (ssize_t)size);
    return 0;
}
