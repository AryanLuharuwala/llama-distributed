#include "gpu_lock.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace dist {

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

} // namespace dist
