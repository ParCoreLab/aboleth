#include "mlx5dv_shim.h"

namespace aboleth::dl {

namespace {

constexpr const char* kDescription = "libmlx5";
const char* const kCandidates[] = {"libmlx5.so.1", "libmlx5.so", nullptr};

Library& library() {
    static Library instance(kCandidates, kDescription);
    return instance;
}

}  // namespace

const Mlx5dvSymbols& mlx5dv() {
    static const Mlx5dvSymbols table = [] {
        library().require();
        Mlx5dvSymbols symbols{};
#define ABOLETH_MLX5DV_RESOLVE(name) \
    resolve(library(), symbols.name, #name, kDescription);
        ABOLETH_MLX5DV_SYMBOLS(ABOLETH_MLX5DV_RESOLVE)
#undef ABOLETH_MLX5DV_RESOLVE
        return symbols;
    }();
    return table;
}

bool mlx5dv_available() noexcept {
    return library().available();
}

namespace {

// Soft-resolves every DEVX symbol; complete() is false when any is missing
// (an old libmlx5), which callers must treat as "DEVX unavailable".
struct DevxTable {
    Mlx5dvDevxSymbols symbols{};
    bool complete{};
};

const DevxTable& devx_table() {
    static const DevxTable table = [] {
        DevxTable result{};
        if (!library().available()) {
            return result;
        }
        result.complete = true;
#define ABOLETH_MLX5DV_DEVX_RESOLVE(name)                              \
    result.symbols.name = reinterpret_cast<decltype(&::name)>(         \
        library().sym(#name));                                         \
    if (result.symbols.name == nullptr) {                              \
        result.complete = false;                                       \
    }
        ABOLETH_MLX5DV_DEVX_SYMBOLS(ABOLETH_MLX5DV_DEVX_RESOLVE)
#undef ABOLETH_MLX5DV_DEVX_RESOLVE
        return result;
    }();
    return table;
}

}  // namespace

const Mlx5dvDevxSymbols& mlx5dv_devx() {
    return devx_table().symbols;
}

bool mlx5dv_devx_available() noexcept {
    return devx_table().complete;
}

}  // namespace aboleth::dl
