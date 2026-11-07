#pragma once

#include "aboleth_dl.h"

#include <infiniband/efadv.h>

namespace aboleth::dl {

// The EFA direct-verbs extension. efadv_query_qp_wqs and efadv_query_cq are the
// calls that hand the GPU-initiated path its queue mappings, and they only
// exist from EFA_1.4 — an older libefa fails the symbol resolution rather than
// silently degrading.
#define ABOLETH_EFADV_SYMBOLS(X) \
    X(efadv_query_device)        \
    X(efadv_create_qp_ex)        \
    X(efadv_create_cq)           \
    X(efadv_query_ah)            \
    X(efadv_query_cq)            \
    X(efadv_query_qp_wqs)

struct EfadvSymbols {
#define ABOLETH_EFADV_DECLARE(name) decltype(&::name) name;
    ABOLETH_EFADV_SYMBOLS(ABOLETH_EFADV_DECLARE)
#undef ABOLETH_EFADV_DECLARE
};

const EfadvSymbols& efadv();
bool efadv_available() noexcept;

}  // namespace aboleth::dl
