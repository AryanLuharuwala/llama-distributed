// gpu_lock.h
//
// Cross-process advisory lock around GPU compute.  Two llama.cpp contexts
// backed by the same CUDA device destabilise each other when they issue
// decodes concurrently (CUDA graph reuse + per-context scratch buffers blow
// up on the second user).  Until per-device isolation lands in the backend,
// every dist-node holding a GPU engine must take this lock for the duration
// of a decode.
//
// The lock is held on a file under /tmp named after the device index, so all
// processes targeting the same GPU see the same lock file.  Flock is advisory
// — non-participants are unaffected.

#pragma once

#include <cstdint>
#include <string>

namespace dist {

class GpuLock {
public:
    // Open (and create if needed) the lock file for `device`.  Returns false
    // on filesystem errors; see last_error().  A device of -1 disables the
    // lock entirely (CPU-only nodes).
    bool open(int device);

    // Acquire the flock (blocking).  No-op when disabled.
    void acquire();

    // Release the flock.
    void release();

    bool enabled() const { return fd_ >= 0; }

    const std::string & last_error() const { return err_; }

    ~GpuLock();

private:
    int fd_ = -1;
    std::string err_;
};

// Scoped helper.  Safe to pass a null pointer — acts as a no-op.
class GpuLockGuard {
public:
    explicit GpuLockGuard(GpuLock * l) : l_(l) { if (l_) l_->acquire(); }
    ~GpuLockGuard() { if (l_) l_->release(); }
    GpuLockGuard(const GpuLockGuard &) = delete;
    GpuLockGuard & operator=(const GpuLockGuard &) = delete;
private:
    GpuLock * l_;
};

} // namespace dist
