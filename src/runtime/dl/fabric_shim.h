#pragma once

#include "aboleth_dl.h"

#include <rdma/fabric.h>

namespace aboleth::dl {

// libfabric's control plane. Everything after fi_fabric() — fi_domain,
// fi_endpoint, fi_write, fi_cq_read, fi_close — is a static inline dispatching
// through the fid_*->ops vtable of an object these calls returned, so the
// dynamic surface stays this small.
#define ABOLETH_FABRIC_SYMBOLS(X) \
    X(fi_getinfo)                 \
    X(fi_dupinfo)                 \
    X(fi_freeinfo)                \
    X(fi_fabric)                  \
    X(fi_strerror)

struct FabricSymbols {
#define ABOLETH_FABRIC_DECLARE(name) decltype(&::name) name;
    ABOLETH_FABRIC_SYMBOLS(ABOLETH_FABRIC_DECLARE)
#undef ABOLETH_FABRIC_DECLARE

    // fabric.h's fi_allocinfo() is a static inline over fi_dupinfo(NULL); going
    // through it directly would leave fi_dupinfo as a link-time reference.
    ::fi_info* allocinfo() const { return fi_dupinfo(nullptr); }
};

const FabricSymbols& fabric();
bool fabric_available() noexcept;

}  // namespace aboleth::dl
