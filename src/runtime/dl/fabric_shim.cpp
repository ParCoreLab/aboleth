#include "fabric_shim.h"

namespace aboleth::dl {

namespace {

constexpr const char* kDescription = "libfabric";
const char* const kCandidates[] = {"libfabric.so.1", "libfabric.so", nullptr};

Library& library() {
    static Library instance(kCandidates, kDescription);
    return instance;
}

}  // namespace

const FabricSymbols& fabric() {
    static const FabricSymbols table = [] {
        library().require();
        FabricSymbols symbols{};
#define ABOLETH_FABRIC_RESOLVE(name) \
    resolve(library(), symbols.name, #name, kDescription);
        ABOLETH_FABRIC_SYMBOLS(ABOLETH_FABRIC_RESOLVE)
#undef ABOLETH_FABRIC_RESOLVE
        return symbols;
    }();
    return table;
}

bool fabric_available() noexcept {
    return library().available();
}

}  // namespace aboleth::dl
