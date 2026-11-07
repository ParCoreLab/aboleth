#include "aboleth_dl.h"

#include <dlfcn.h>

#include <mutex>

namespace aboleth::dl {

namespace {

// dlopen/dlerror share process-wide state, and providers are opened from
// whichever thread first touches them.
std::mutex& loader_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

Library::Library(
    const char* const* candidates,
    const char* description) noexcept
    : candidates_(candidates), description_(description) {}

void Library::load() noexcept {
    if (attempted_) {
        return;
    }
    attempted_ = true;

    // RTLD_LOCAL: the provider is a private implementation detail, and putting
    // its symbols in the global namespace could shadow a differently-versioned
    // copy the host application already loaded.
    for (const char* const* name = candidates_; *name != nullptr; ++name) {
        handle_ = dlopen(*name, RTLD_NOW | RTLD_LOCAL);
        if (handle_ != nullptr) {
            return;
        }
        const char* message = dlerror();
        if (!error_.empty()) {
            error_ += "; ";
        }
        error_ += message != nullptr ? message : *name;
    }
}

bool Library::available() noexcept {
    std::lock_guard<std::mutex> guard(loader_mutex());
    load();
    return handle_ != nullptr;
}

const std::string& Library::error() noexcept {
    std::lock_guard<std::mutex> guard(loader_mutex());
    load();
    return error_;
}

void Library::require() {
    if (!available()) {
        throw_unavailable(description_, error());
    }
}

void* Library::sym(const char* name) noexcept {
    std::lock_guard<std::mutex> guard(loader_mutex());
    load();
    if (handle_ == nullptr) {
        return nullptr;
    }
    dlerror();
    return dlsym(handle_, name);
}

void throw_unavailable(
    const char* description,
    const std::string& detail) {
    std::string message = description;
    message += " is not available on this host";
    if (!detail.empty()) {
        message += ": " + detail;
    }
    throw std::runtime_error(message);
}

void throw_missing_symbol(const char* description, const char* symbol) {
    std::string message = description;
    message += " on this host does not export ";
    message += symbol;
    throw std::runtime_error(message);
}

}  // namespace aboleth::dl
