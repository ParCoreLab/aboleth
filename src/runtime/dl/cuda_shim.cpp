#include "cuda_shim.h"

namespace aboleth::dl {

namespace {

constexpr const char* kDescription = "libcuda (the NVIDIA driver)";
const char* const kCandidates[] = {"libcuda.so.1", "libcuda.so", nullptr};

Library& library() {
    static Library instance(kCandidates, kDescription);
    return instance;
}

}  // namespace

const CudaSymbols& cuda() {
    static const CudaSymbols table = [] {
        library().require();
        CudaSymbols symbols{};
#define ABOLETH_CUDA_RESOLVE(name) \
    resolve(library(), symbols.name, #name, kDescription);
        ABOLETH_CUDA_SYMBOLS(ABOLETH_CUDA_RESOLVE)
#undef ABOLETH_CUDA_RESOLVE
        return symbols;
    }();
    return table;
}

bool cuda_available() noexcept {
    return library().available();
}

}  // namespace aboleth::dl
