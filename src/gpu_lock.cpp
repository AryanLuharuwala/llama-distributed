#include "gpu_lock.h"
#include "platform_compat.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

namespace dist {

#ifdef _WIN32

// On Windows we use LockFileEx on an empty file under %TEMP%.  `fd_` holds
// the HANDLE cast to intptr_t; -1 means "disabled".  LockFileEx operates
// on a byte range, so we always lock the first byte.

bool GpuLock::open(int device) {
    if (device < 0) { fd_ = -1; return true; }

    char buf[MAX_PATH];
    DWORD n = ::GetTempPathA(MAX_PATH, buf);
    if (n == 0 || n >= MAX_PATH) {
        err_ = "GetTempPath failed";
        fd_ = -1;
        return false;
    }
    std::string path(buf, n);
    char tail[64];
    std::snprintf(tail, sizeof(tail), "dist-gpu-%d.lock", device);
    path += tail;

    HANDLE h = ::CreateFileA(path.c_str(),
                             GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr,
                             OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err_ = "CreateFile " + path + " failed: " + std::to_string(::GetLastError());
        fd_ = -1;
        return false;
    }
    fd_ = (int)(intptr_t)h;
    return true;
}

void GpuLock::acquire() {
    if (fd_ < 0) return;
    HANDLE h = (HANDLE)(intptr_t)fd_;
    OVERLAPPED ov{};
    // Blocking exclusive lock on the first byte.
    ::LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &ov);
}

void GpuLock::release() {
    if (fd_ < 0) return;
    HANDLE h = (HANDLE)(intptr_t)fd_;
    OVERLAPPED ov{};
    ::UnlockFileEx(h, 0, 1, 0, &ov);
}

GpuLock::~GpuLock() {
    if (fd_ >= 0) {
        HANDLE h = (HANDLE)(intptr_t)fd_;
        ::CloseHandle(h);
    }
}

#else  // ── POSIX ─────────────────────────────────────────────────────────

bool GpuLock::open(int device) {
    if (device < 0) {
        fd_ = -1;
        return true;
    }
    char path[128];
    std::snprintf(path, sizeof(path), "/tmp/dist-gpu-%d.lock", device);
    fd_ = ::open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) {
        err_ = std::string("open ") + path + ": " + std::strerror(errno);
        return false;
    }
    return true;
}

void GpuLock::acquire() {
    if (fd_ < 0) return;
    while (::flock(fd_, LOCK_EX) < 0) {
        if (errno == EINTR) continue;
        break; // best-effort; failures are surfaced via err_ if needed
    }
}

void GpuLock::release() {
    if (fd_ < 0) return;
    ::flock(fd_, LOCK_UN);
}

GpuLock::~GpuLock() {
    if (fd_ >= 0) ::close(fd_);
}

#endif

} // namespace dist
