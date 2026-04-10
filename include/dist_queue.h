#pragma once
/**
 * dist_queue.h
 *
 * Bounded, thread-safe MPSC/MPMC queue used to pipeline:
 *   - incoming tensor batches (network thread -> compute thread)
 *   - outgoing tensor batches (compute thread -> network thread)
 */

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <chrono>

namespace dist {

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity = 4) : cap_(capacity) {}

    // Push item. Blocks if full. Returns false if closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        not_full_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
        if (closed_) return false;
        q_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Try push without blocking. Returns false if full or closed.
    bool try_push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        if (closed_ || q_.size() >= cap_) return false;
        q_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Pop item. Blocks until available or closed.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        T item = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return item;
    }

    // Pop with timeout. Returns nullopt on timeout or close.
    std::optional<T> pop_timeout(std::chrono::milliseconds ms) {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait_for(lk, ms, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        T item = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return item;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lk(mu_);
        return closed_;
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T>           q_;
    size_t                  cap_;
    bool                    closed_ = false;
};

} // namespace dist
