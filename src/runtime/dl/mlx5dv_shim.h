#pragma once

#include "aboleth_dl.h"

#include <infiniband/mlx5dv.h>

namespace aboleth::dl {

// mlx5dv_init_obj is the only mlx5 direct-verbs export the datapath calls; the
// queue accessors it hands back are plain memory, and the doorbell path is
// inline assembly over that memory. mlx5dv_is_supported exists for device
// selection: udev may rename an mlx5 HCA to anything, so capability is asked
// of the driver rather than read out of the device name.
#define ABOLETH_MLX5DV_SYMBOLS(X) \
    X(mlx5dv_init_obj) \
    X(mlx5dv_is_supported)

struct Mlx5dvSymbols {
#define ABOLETH_MLX5DV_DECLARE(name) decltype(&::name) name;
    ABOLETH_MLX5DV_SYMBOLS(ABOLETH_MLX5DV_DECLARE)
#undef ABOLETH_MLX5DV_DECLARE
};

const Mlx5dvSymbols& mlx5dv();
bool mlx5dv_available() noexcept;

// The DEVX object surface the GPU-resident staged endpoint uses. Resolved
// separately and OPTIONALLY: an older libmlx5 without these exports must not
// take down the classic endpoint paths, so absence is reported through
// mlx5dv_devx_available() rather than thrown from the resolver.
#define ABOLETH_MLX5DV_DEVX_SYMBOLS(X) \
    X(mlx5dv_devx_obj_create) \
    X(mlx5dv_devx_obj_modify) \
    X(mlx5dv_devx_obj_destroy) \
    X(mlx5dv_devx_umem_reg) \
    X(mlx5dv_devx_umem_reg_ex) \
    X(mlx5dv_devx_umem_dereg) \
    X(mlx5dv_devx_alloc_uar) \
    X(mlx5dv_devx_free_uar) \
    X(mlx5dv_devx_query_eqn) \
    X(mlx5dv_devx_obj_query) \
    X(mlx5dv_devx_qp_query)

struct Mlx5dvDevxSymbols {
#define ABOLETH_MLX5DV_DEVX_DECLARE(name) decltype(&::name) name;
    ABOLETH_MLX5DV_DEVX_SYMBOLS(ABOLETH_MLX5DV_DEVX_DECLARE)
#undef ABOLETH_MLX5DV_DEVX_DECLARE
};

const Mlx5dvDevxSymbols& mlx5dv_devx();
bool mlx5dv_devx_available() noexcept;

}  // namespace aboleth::dl
