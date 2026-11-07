#pragma once

#include "aboleth_dl.h"

#include <cuda.h>

namespace aboleth::dl {

// The CUDA *driver* API. libcudart stays a normal link dependency — it dlopens
// libcuda.so.1 itself and degrades on its own — but the driver entry points the
// runtime calls directly for virtual-memory management and stream-ordered
// writes are resolved here, so a machine with no NVIDIA driver installed can
// still load the library and report no providers.
//
// The member names carry the _v2 suffixes because cuda.h renames those entry
// points with object-like macros; keeping the suffix is what lets a call site
// stay spelled cuMemHostRegister(...) and still expand onto the right member.
#define ABOLETH_CUDA_SYMBOLS(X)        \
    X(cuGetErrorName)                  \
    X(cuGetErrorString)                \
    X(cuCtxGetDevice)                  \
    X(cuDeviceGetAttribute)            \
    X(cuMemHostRegister_v2)            \
    X(cuMemHostUnregister)             \
    X(cuMemHostGetDevicePointer_v2)    \
    X(cuMemAddressReserve)             \
    X(cuMemAddressFree)                \
    X(cuMemCreate)                     \
    X(cuMemRelease)                    \
    X(cuMemMap)                        \
    X(cuMemUnmap)                      \
    X(cuMemSetAccess)                  \
    X(cuMemGetAllocationGranularity)   \
    X(cuMemGetHandleForAddressRange)   \
    X(cuPointerGetAttribute)           \
    X(cuPointerSetAttribute)           \
    X(cuStreamWriteValue32_v2)

struct CudaSymbols {
#define ABOLETH_CUDA_DECLARE(name) decltype(&::name) name;
    ABOLETH_CUDA_SYMBOLS(ABOLETH_CUDA_DECLARE)
#undef ABOLETH_CUDA_DECLARE
};

const CudaSymbols& cuda();
bool cuda_available() noexcept;

}  // namespace aboleth::dl
