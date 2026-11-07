#pragma once

// Runtime resolution of the transport libraries.
//
// libaboleth_rdma_abi.so carries every provider it was compiled with, but a
// machine only ever has some of the hardware. If the transports were ordinary
// DT_NEEDED dependencies, one absent library — no EFA adapter, no mlx5, no
// libfabric — would make the whole runtime, and therefore `import aboleth`,
// fail to load. Resolving them with dlopen instead lets one binary run
// anywhere, reporting through aboleth_runtime_provider_caps() which providers
// are actually usable.
//
// Only entry points that are real exported symbols need an entry here. The hot
// path does not: ibv_post_send, ibv_poll_cq, the mlx5dv accessors, and the fi_*
// data-path calls are all static inlines in the headers, dispatching through an
// ops vtable reached from an object the control plane already created.

#include <stdexcept>
#include <string>

namespace aboleth::dl {

// A lazily dlopen'd shared library. Loading is attempted exactly once per
// process; a failed load is remembered, not retried.
class Library {
public:
    // `candidates` is a null-terminated list of sonames tried in order, most
    // specific first (libibverbs.so.1 before libibverbs.so).
    Library(const char* const* candidates, const char* description) noexcept;

    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    // True when the library loaded. Never throws; this is what the capability
    // probes call.
    bool available() noexcept;

    // Throws if the library is unavailable, naming it and the dlerror text.
    void require();

    // Returns nullptr when the symbol is absent, which is how an older library
    // version is detected.
    void* sym(const char* name) noexcept;

    const std::string& error() noexcept;

private:
    void load() noexcept;

    const char* const* candidates_;
    const char* description_;
    void* handle_ = nullptr;
    bool attempted_ = false;
    std::string error_;
};

[[noreturn]] void throw_unavailable(
    const char* description,
    const std::string& detail);

[[noreturn]] void throw_missing_symbol(
    const char* description,
    const char* symbol);

// Fills one slot of a symbol table. The slot type comes from the library's own
// header via decltype(&::symbol), so the table cannot drift from the ABI it
// describes.
template <typename Slot>
void resolve(
    Library& library,
    Slot& slot,
    const char* symbol,
    const char* description) {
    void* address = library.sym(symbol);
    if (address == nullptr) {
        throw_missing_symbol(description, symbol);
    }
    slot = reinterpret_cast<Slot>(address);
}

}  // namespace aboleth::dl
