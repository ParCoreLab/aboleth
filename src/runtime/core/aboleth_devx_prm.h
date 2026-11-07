#pragma once

// PRM/DEVX plumbing for the GPU-resident staged endpoint.
//
// The bit-exact mailbox layouts come from the vendored headers in
// src/runtime/rdma/vendor (never re-typed by hand — see the README there);
// this header adds the DEVX_SET/DEVX_GET accessor layer (the same macro
// structure NVSHMEM's IBGDA transport and rdma-core's provider use) and the
// handful of qpc/cqc field values the PRM enums name.

#include <endian.h>
#include <stddef.h>
#include <stdint.h>

#include "../rdma/vendor/mlx5_ifc.h"
#include "../rdma/vendor/mlx5_prm.h"

#ifndef DEVX_SET

#define __devx_nullp(typ) ((struct mlx5_ifc_##typ##_bits *)0)
#define __devx_st_sz_bits(typ) sizeof(struct mlx5_ifc_##typ##_bits)
#define DEVX_ST_SZ_BYTES(typ) (__devx_st_sz_bits(typ) / 8)
#define DEVX_ST_SZ_DW(typ)    (__devx_st_sz_bits(typ) / 32)

#define __devx_bit_sz(typ, fld)  sizeof(__devx_nullp(typ)->fld)
#define __devx_bit_off(typ, fld) offsetof(struct mlx5_ifc_##typ##_bits, fld)
#define __devx_dw_off(typ, fld)  (__devx_bit_off(typ, fld) / 32)
#define __devx_dw_bit_off(typ, fld) \
    (32 - __devx_bit_sz(typ, fld) - (__devx_bit_off(typ, fld) & 0x1f))
#define __devx_mask(typ, fld) \
    ((uint32_t)((1ull << __devx_bit_sz(typ, fld)) - 1))
#define __devx_dw_mask(typ, fld) \
    (__devx_mask(typ, fld) << __devx_dw_bit_off(typ, fld))

#define DEVX_SET(typ, p, fld, v)                                          \
    do {                                                                  \
        uint32_t *_p = (uint32_t *)(p) + __devx_dw_off(typ, fld);         \
        uint32_t _dw = be32toh(*_p);                                      \
        _dw &= ~__devx_dw_mask(typ, fld);                                 \
        _dw |= (((uint32_t)(v) & __devx_mask(typ, fld))                   \
                << __devx_dw_bit_off(typ, fld));                          \
        *_p = htobe32(_dw);                                               \
    } while (0)

#define DEVX_GET(typ, p, fld)                                             \
    ((be32toh(*((uint32_t *)(p) + __devx_dw_off(typ, fld)))               \
      >> __devx_dw_bit_off(typ, fld)) & __devx_mask(typ, fld))

/* 64-bit PRM fields are 64-bit aligned. */
#define DEVX_SET64(typ, p, fld, v)                                        \
    do {                                                                  \
        *((uint64_t *)((uint8_t *)(p) + __devx_bit_off(typ, fld) / 8)) =  \
            htobe64((uint64_t)(v));                                       \
    } while (0)

#define DEVX_ADDR_OF(typ, p, fld)                                         \
    ((void *)((uint8_t *)(p) + __devx_bit_off(typ, fld) / 8))

#endif /* DEVX_SET */

// qpc field values (PRM enums; guarded, the vendored header may name some).
#ifndef MLX5_QPC_PM_STATE_MIGRATED
#define MLX5_QPC_PM_STATE_MIGRATED 0x3
#endif
// qpc.rq_type: 0=REGULAR 1=SRQ 2=CRQ 3=ZERO_SIZE_RQ. A send-only QP uses
// ZERO_SIZE_RQ so the SQ sits at wq-umem offset 0 (a REGULAR RQ would put
// the RQ first in the buffer, rdma-core's layout).
#ifndef MLX5_QPC_RQ_TYPE_ZERO_SIZE_RQ
#define MLX5_QPC_RQ_TYPE_ZERO_SIZE_RQ 0x3
#endif
// qpc.atomic_mode: NVSHMEM ibgda.cpp's UP_TO_64BIT — required for the
// remote-atomic signal path.
#ifndef ABOLETH_QPC_ATOMIC_MODE_UP_TO_64BIT
#define ABOLETH_QPC_ATOMIC_MODE_UP_TO_64BIT 0x3
#endif
