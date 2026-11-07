#pragma once

// Convenience aggregate: the provider symbol tables the runtime translation
// units call. Each shim header also pulls in the corresponding vendor header,
// so including this replaces the direct <infiniband/...> and <rdma/...>
// includes at the top of a runtime source.

#include "aboleth_dl.h"

#if ABOLETH_HAS_VERBS
#include "verbs_shim.h"
#endif
#if ABOLETH_HAS_MLX5
#include "mlx5dv_shim.h"
#endif
#if ABOLETH_HAS_EFA
#include "efadv_shim.h"
#endif
#if ABOLETH_HAS_LIBFABRIC
#include "fabric_shim.h"
#endif
#if ABOLETH_HAS_CUDA
#include "cuda_shim.h"
#endif
