#include "aboleth/runtime_c.h"

#include "dl/shims.h"

namespace {

// A provider that resolved its library but has no device is worse than useless
// to a planner: it would be chosen and then fail at endpoint creation.
bool has_verbs_device(bool (*matches)(const char*)) {
#if ABOLETH_HAS_VERBS
    int count = 0;
    ::ibv_device** devices = aboleth::dl::verbs().ibv_get_device_list(&count);
    if (devices == nullptr) {
        return false;
    }
    bool found = false;
    for (int index = 0; index < count && !found; ++index) {
        const char* name = aboleth::dl::verbs().ibv_get_device_name(devices[index]);
        found = name != nullptr && (matches == nullptr || matches(name));
    }
    aboleth::dl::verbs().ibv_free_device_list(devices);
    return found;
#else
    (void)matches;
    return false;
#endif
}

#if ABOLETH_HAS_EFA
bool is_efa_device(const char* name) {
    return name[0] == 'e' && name[1] == 'f' && name[2] == 'a';
}
#endif

}  // namespace

extern "C" std::uint32_t aboleth_runtime_provider_caps() {
    std::uint32_t caps = 0;

#if ABOLETH_HAS_CUDA
    if (aboleth::dl::cuda_available()) {
        caps |= kAbolethProviderCuda;
    }
#endif
#if ABOLETH_HAS_LIBFABRIC
    if (aboleth::dl::fabric_available()) {
        caps |= kAbolethProviderLibfabric;
    }
#endif
#if ABOLETH_HAS_VERBS
    if (aboleth::dl::verbs_available() && has_verbs_device(nullptr)) {
        caps |= kAbolethProviderVerbs;
    }
#endif
#if ABOLETH_HAS_MLX5
    if ((caps & kAbolethProviderVerbs) != 0 && aboleth::dl::mlx5dv_available()) {
        caps |= kAbolethProviderMlx5;
    }
#endif
#if ABOLETH_HAS_EFA
    if ((caps & kAbolethProviderVerbs) != 0
        && (caps & kAbolethProviderLibfabric) != 0
        && aboleth::dl::efadv_available()
        && has_verbs_device(is_efa_device)) {
        caps |= kAbolethProviderEfa;
    }
#endif
    return caps;
}
