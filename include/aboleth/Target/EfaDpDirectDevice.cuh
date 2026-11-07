#pragma once

// efa-dp-direct declares its device primitives with external linkage and then
// defines them as static inline in the implementation header. CUDA 13 rejects
// that mismatch. Import the stable type/enum surface directly and suppress the
// conflicting declarations.
#ifndef EFA_CUDA_DP_CUH
#define EFA_CUDA_DP_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "efa_cuda_dp_types.h"
#include "efa_io_defs.h"

enum efa_cuda_wc_opcode {
    EFA_CUDA_WC_SEND,
    EFA_CUDA_WC_RDMA_WRITE,
    EFA_CUDA_WC_RDMA_READ,
    EFA_CUDA_WC_RECV = 1 << 7,
    EFA_CUDA_WC_RECV_RDMA_WITH_IMM,
};

enum efa_cuda_processing_hint {
    EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE = 1 << 0,
};

#endif

#include "efa_cuda_dp_impl.cuh"
