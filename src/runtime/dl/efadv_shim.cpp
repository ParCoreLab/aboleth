#include "efadv_shim.h"

namespace aboleth::dl {

namespace {

constexpr const char* kDescription = "libefa";
const char* const kCandidates[] = {"libefa.so.1", "libefa.so", nullptr};

Library& library() {
    static Library instance(kCandidates, kDescription);
    return instance;
}

}  // namespace

const EfadvSymbols& efadv() {
    static const EfadvSymbols table = [] {
        library().require();
        EfadvSymbols symbols{};
#define ABOLETH_EFADV_RESOLVE(name) \
    resolve(library(), symbols.name, #name, kDescription);
        ABOLETH_EFADV_SYMBOLS(ABOLETH_EFADV_RESOLVE)
#undef ABOLETH_EFADV_RESOLVE
        return symbols;
    }();
    return table;
}

bool efadv_available() noexcept {
    return library().available();
}

}  // namespace aboleth::dl
