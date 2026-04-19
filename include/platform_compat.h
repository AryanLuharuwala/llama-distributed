// platform_compat.h
//
// Single include to paper over POSIX / Win32 differences for the bits we
// actually use: BSD sockets, file permissions, sleep, and file locking.
//
// Rules:
//   - Include this instead of <arpa/inet.h>, <sys/socket.h>, <netdb.h>,
//     <netinet/tcp.h>, <unistd.h>, <sys/file.h>, <sys/stat.h>.
//   - Use `dist::close_sock(fd)` to close a socket (don't call `::close`).
//   - Use `dist::chmod_0600(path)` when you'd write `chmod(path, 0600)`.
//   - Call `dist::net_startup()` once at main() entry.  Harmless on POSIX.
//   - `MSG_NOSIGNAL` is a no-op on Windows; use the macro either way.
//
// This header intentionally stays small.  If you need a Win32 API that
// isn't here, add it here rather than sprinkling #ifdefs across .cpp files.

#pragma once

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  include <io.h>
#  include <process.h>
#  pragma comment(lib, "Ws2_32.lib")

   // MSVC doesn't define ssize_t.
#  include <BaseTsd.h>
   using ssize_t = SSIZE_T;

   // Winsock accepts send() with a plain 0 flag; there's no SIGPIPE to
   // mask.  Define MSG_NOSIGNAL as 0 so call sites compile unchanged.
#  ifndef MSG_NOSIGNAL
#    define MSG_NOSIGNAL 0
#  endif

   // Winsock doesn't ship SHUT_* constants — provide the usual names.
#  ifndef SHUT_RD
#    define SHUT_RD   SD_RECEIVE
#    define SHUT_WR   SD_SEND
#    define SHUT_RDWR SD_BOTH
#  endif

#else  // ── POSIX ──────────────────────────────────────────────────────────
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/file.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/time.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

#include <cstdint>
#include <string>

namespace dist {

#ifdef _WIN32
using sock_t = SOCKET;
inline constexpr sock_t SOCK_INVALID_V = INVALID_SOCKET;
#else
using sock_t = int;
inline constexpr sock_t SOCK_INVALID_V = -1;
#endif

// One-line helpers — inline so they don't need a .cpp.

inline int close_sock(sock_t s) {
#ifdef _WIN32
    return ::closesocket(s);
#else
    return ::close(s);
#endif
}

// Owner-only (0600) permissions on a regular file.  No-op on Windows where
// the notion doesn't map cleanly — the file inherits its parent ACL, which
// is fine for the places we use this (agent.env, token stores in %APPDATA%).
inline int chmod_0600(const std::string & path) {
#ifdef _WIN32
    (void)path;
    return 0;
#else
    return ::chmod(path.c_str(), 0600);
#endif
}

inline void sleep_ms(uint32_t ms) {
#ifdef _WIN32
    ::Sleep(ms);
#else
    ::usleep(ms * 1000u);
#endif
}

// Initialise the sockets subsystem.  WSAStartup on Windows, no-op on POSIX.
// Safe to call more than once; returns false only on Windows if startup
// fails outright (in which case the caller should bail).
inline bool net_startup() {
#ifdef _WIN32
    static bool done = false;
    if (done) return true;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    done = true;
    return true;
#else
    return true;
#endif
}

} // namespace dist
