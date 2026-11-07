// Opening a libfabric fabric on an EFA device.
//
// The whole of the EFA control plane above the endpoints: one fabric, one
// domain and one shared address vector per opened device, plus the
// FI_EFA_GDA_OPS handle an efa-direct runtime hands the GPU-initiated path.
// Every EFA endpoint in the runtime — proxy, GDA and Transport.EFA alike —
// hangs off the context this file builds; the types it fills in are declared
// in aboleth_efa_internal.h.

#include "aboleth_efa_internal.h"

#if ABOLETH_HAS_EFA

#include "../aboleth_runtime_common.h"
#include "../aboleth_runtime_internal.h"

#include "../dl/shims.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_ext_efa.h>
#include <rdma/fi_rma.h>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace dl = aboleth::dl;

namespace aboleth::runtime {

std::shared_ptr<EfaFabricContext>
create_efa_fabric_context(
    int gpu_index,
    const char* requested_name) {
    if (setenv(
            "FI_EFA_ENABLE_SHM_TRANSFER",
            "0",
            1) != 0) {
        throw_error(
            "failed to disable EFA shared-memory transfers");
    }
    auto context =
        std::make_shared<EfaFabricContext>();
    context->device_name =
        select_efa_device_name(
            gpu_index,
            requested_name);
    context->domain_name =
        context->device_name + "-rdm";
    const char* fabric_name =
        std::getenv("ABOLETH_EFA_FABRIC");
    context->fabric_name =
        fabric_name != nullptr &&
                fabric_name[0] != '\0'
            ? fabric_name
            : "efa";
    if (context->fabric_name != "efa" &&
        context->fabric_name !=
            "efa-direct") {
        throw_error(
            "ABOLETH_EFA_FABRIC must be efa or efa-direct");
    }
    const bool direct =
        context->fabric_name ==
        "efa-direct";

    fi_info* hints = dl::fabric().allocinfo();
    if (hints == nullptr) {
        throw_error("fi_allocinfo failed");
    }
    hints->caps =
        FI_RMA |
        FI_WRITE |
        FI_REMOTE_WRITE |
        FI_HMEM;
    if (!direct) {
        hints->caps |=
            FI_LOCAL_COMM |
            FI_REMOTE_COMM;
    }
    if (direct) {
        hints->mode = FI_CONTEXT2;
    }
    hints->ep_attr->type = FI_EP_RDM;
    hints->tx_attr->tclass =
        FI_TC_LOW_LATENCY;
    hints->domain_attr->threading =
        FI_THREAD_SAFE;
    hints->domain_attr->data_progress =
        direct
            ? FI_PROGRESS_AUTO
            : FI_PROGRESS_MANUAL;
    hints->domain_attr->control_progress =
        direct
            ? FI_PROGRESS_AUTO
            : FI_PROGRESS_MANUAL;
    hints->domain_attr->av_type =
        FI_AV_TABLE;
    hints->domain_attr->mr_mode =
        FI_MR_LOCAL |
        FI_MR_VIRT_ADDR |
        FI_MR_ALLOCATED |
        FI_MR_PROV_KEY |
        FI_MR_HMEM;
    hints->fabric_attr->prov_name =
        strdup("efa");
    hints->fabric_attr->name =
        strdup(
            context->fabric_name.c_str());
    hints->domain_attr->name =
        strdup(context->domain_name.c_str());

    const int getinfo_status =
        dl::fabric().fi_getinfo(
            FI_VERSION(2, 4),
            nullptr,
            nullptr,
            0,
            hints,
            &context->info);
    dl::fabric().fi_freeinfo(hints);
    fabric_check(
        getinfo_status,
        "fi_getinfo(EFA)");
    if (context->info == nullptr ||
        context->info->addr_format !=
            FI_ADDR_EFA) {
        throw_error(
            "EFA provider did not return an EFA address");
    }
    if ((context->info->caps & FI_HMEM) == 0 ||
        (context->info->caps & FI_RMA) == 0) {
        throw_error(
            "EFA provider lacks CUDA HMEM RMA support");
    }
    if (context->info->domain_attr->mr_key_size >
        sizeof(std::uint32_t)) {
        throw_error(
            "EFA memory keys exceed the proxy ABI width");
    }
    fabric_check(
        dl::fabric().fi_fabric(
            context->info->fabric_attr,
            &context->fabric,
            nullptr),
        "fi_fabric(EFA)");
    fabric_check(
        fi_domain(
            context->fabric,
            context->info,
            &context->domain,
            nullptr),
        "fi_domain(EFA)");
    if (direct) {
        fabric_check(
            fi_open_ops(
                &context->domain->fid,
                FI_EFA_GDA_OPS,
                0,
                reinterpret_cast<void**>(
                    &context->gda_ops),
                nullptr),
            "fi_open_ops(EFA GDA)");
        if (context->gda_ops == nullptr) {
            throw_error(
                "EFA GDA operations are unavailable");
        }
    }
    fi_av_attr av_attributes{};
    av_attributes.type = FI_AV_TABLE;
    av_attributes.count = 4096;
    fabric_check(
        fi_av_open(
            context->domain,
            &av_attributes,
            &context->proxy_av,
            nullptr),
        "fi_av_open(EFA shared proxy)");
    return context;
}

}  // namespace aboleth::runtime

#endif  // ABOLETH_HAS_EFA
