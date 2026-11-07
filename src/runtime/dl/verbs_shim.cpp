#include "verbs_shim.h"

namespace aboleth::dl {

namespace {

constexpr const char* kDescription = "libibverbs";
const char* const kCandidates[] = {"libibverbs.so.1", "libibverbs.so", nullptr};

Library& library() {
    static Library instance(kCandidates, kDescription);
    return instance;
}

}  // namespace

const VerbsSymbols& verbs() {
    static const VerbsSymbols table = [] {
        library().require();
        VerbsSymbols symbols{};
#define ABOLETH_VERBS_RESOLVE(name) \
    resolve(library(), symbols.name, #name, kDescription);
        ABOLETH_VERBS_SYMBOLS(ABOLETH_VERBS_RESOLVE)
#undef ABOLETH_VERBS_RESOLVE
        return symbols;
    }();
    return table;
}

bool verbs_available() noexcept {
    return library().available();
}

}  // namespace aboleth::dl
