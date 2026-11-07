#pragma once

#include "aboleth_dl.h"

#include <infiniband/verbs.h>

#include <cstring>
#include <cstdint>

// verbs.h hides two of these entry points behind function-like macros —
// ibv_reg_mr picks between the IBVERBS_1.1 symbol and ibv_reg_mr_iova2 by access
// flags, ibv_query_port prefers the verbs_context op over the compat symbol.
// Left in place the macros would swallow the `verbs().ibv_...(...)` member
// access below, so they are undefined and reproduced as member functions.
#undef ibv_reg_mr
#undef ibv_reg_mr_iova
#undef ibv_query_port

namespace aboleth::dl {

// Every entry point the control plane calls. The data path (ibv_post_send,
// ibv_poll_cq, ibv_post_recv) is absent on purpose: those are static inlines
// over context->ops and need no dynamic symbol.
#define ABOLETH_VERBS_SYMBOLS(X)  \
    X(ibv_get_device_list)        \
    X(ibv_free_device_list)       \
    X(ibv_get_device_name)        \
    X(ibv_open_device)            \
    X(ibv_close_device)           \
    X(ibv_query_device)           \
    X(ibv_query_port)             \
    X(ibv_query_gid)              \
    X(_ibv_query_gid_ex)          \
    X(ibv_alloc_pd)               \
    X(ibv_dealloc_pd)             \
    X(ibv_reg_mr)                 \
    X(ibv_reg_mr_iova2)           \
    X(ibv_reg_dmabuf_mr)          \
    X(ibv_dereg_mr)               \
    X(ibv_create_cq)              \
    X(ibv_destroy_cq)             \
    X(ibv_create_qp)              \
    X(ibv_modify_qp)              \
    X(ibv_destroy_qp)             \
    X(ibv_create_ah)              \
    X(ibv_destroy_ah)

struct VerbsSymbols {
#define ABOLETH_VERBS_DECLARE(name) decltype(&::name) name;
    ABOLETH_VERBS_SYMBOLS(ABOLETH_VERBS_DECLARE)
#undef ABOLETH_VERBS_DECLARE

    // The two call sites that verbs.h does not expose as a plain function.
    ::ibv_mr* reg_mr(
        ::ibv_pd* pd,
        void* addr,
        std::size_t length,
        unsigned int access) const {
        // Matches rdma-core's __ibv_reg_mr: the 1.1 entry point cannot carry
        // the optional access bits (relaxed ordering among them).
        if ((access & IBV_ACCESS_OPTIONAL_RANGE) == 0) {
            return ibv_reg_mr(pd, addr, length, static_cast<int>(access));
        }
        return ibv_reg_mr_iova2(
            pd, addr, length, reinterpret_cast<std::uintptr_t>(addr), access);
    }

    int query_port(
        ::ibv_context* context,
        std::uint8_t port_num,
        ::ibv_port_attr* port_attr) const {
        // Mirrors verbs.h's ___ibv_query_port. Current providers answer through
        // the verbs_context op — a pointer inside an object libibverbs already
        // built, so no symbol — and only the compat path needs the export.
        ::verbs_context* vctx = verbs_get_ctx_op(context, query_port);
        if (vctx == nullptr) {
            std::memset(port_attr, 0, sizeof(*port_attr));
            return ibv_query_port(
                context, port_num,
                reinterpret_cast<::_compat_ibv_port_attr*>(port_attr));
        }
        return vctx->query_port(
            context, port_num, port_attr, sizeof(*port_attr));
    }

    int query_gid_ex(
        ::ibv_context* context,
        std::uint32_t port_num,
        std::uint32_t gid_index,
        ::ibv_gid_entry* entry,
        std::uint32_t flags) const {
        return _ibv_query_gid_ex(
            context, port_num, gid_index, entry, flags, sizeof(*entry));
    }
};

// Resolves libibverbs on first use; throws if it or any entry point is absent.
const VerbsSymbols& verbs();

// Never throws: this is the capability probe.
bool verbs_available() noexcept;

}  // namespace aboleth::dl
