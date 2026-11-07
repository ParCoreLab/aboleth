from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from ..transport_binding import TransportProgramBinding


_CUDA_TYPES = {
    "i1": "bool",
    "i32": "std::uint32_t",
    "i64": "std::uint64_t",
}


def _identifier(value: str) -> str:
    words = [
        word
        for word in re.split(r"[^A-Za-z0-9]+", value)
        if word
    ]
    result = "".join(word[:1].upper() + word[1:] for word in words)
    if not result or result[0].isdigit():
        result = f"Program{result}"
    return result


def binding_signature(binding: TransportProgramBinding) -> str:
    payload = {
        "name": binding.name,
        "operations": binding.operations,
        "kernel": binding.kernel,
        "insertion_point": binding.insertion_point,
        "slots": tuple(
            (slot.name, slot.mlir_type)
            for slot in binding.slots()
        ),
        "regions": tuple(
            {
                "operations": region.operations,
                "remote": (
                    region.remote.pointer.name,
                    region.remote.key.name,
                ),
                "local": (
                    region.local.pointer.name,
                    region.local.key.name,
                ),
                "byte_count": region.byte_count.name,
                "predicate": region.predicate.name,
                "max_bytes": region.max_bytes,
            }
            for region in binding.regions
        ),
        "publication": {
            "remote": (
                binding.publication.remote.pointer.name,
                binding.publication.remote.key.name,
            ),
            "local": (
                binding.publication.local.pointer.name,
                binding.publication.local.key.name,
            ),
            "iteration": binding.publication.iteration.name,
            "iterations": binding.publication.iterations.name,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CudaFragmentArtifact:
    program_name: str
    signature: str
    struct_name: str
    function_name: str
    command_count: int
    source: str


def emit_cuda_fragment(
    binding: TransportProgramBinding,
) -> CudaFragmentArtifact:
    """Emit one planned-mlx5 CUDA fragment from a typed program binding."""
    program = binding.to_program()
    signature = binding_signature(binding)
    stem = _identifier(binding.name)
    struct_name = f"{stem}Bindings"
    function_name = f"emit{stem}PlannedMlx5"
    fields = "\n".join(
        f"    {_CUDA_TYPES[slot.mlir_type]} {slot.name};"
        for slot in binding.slots()
    )
    transfer_lines = []
    for index, region in enumerate(binding.regions):
        transfer_lines.extend((
            f"    if (bindings.{region.predicate.name})",
            "        rdma::write_rdma(",
            f"            endpoint, base + {index},",
            (
                "            reinterpret_cast<const void*>("
                f"bindings.{region.local.pointer.name}),"
            ),
            (
                f"            bindings.{region.local.key.name}, "
                f"bindings.{region.remote.pointer.name},"
            ),
            (
                f"            bindings.{region.remote.key.name}, "
                f"bindings.{region.byte_count.name},"
            ),
            "            /*request_completion=*/false);",
            "    else",
            f"        rdma::write_nop(endpoint, base + {index});",
        ))
    transfer_body = "\n".join(transfer_lines)
    publication = binding.publication
    signal_index = len(binding.regions)
    command_count = program.command_count
    source = f"""#pragma once

#include <cstdint>

#include "aboleth/Target/RdmaAbi.h"
#include "aboleth/Target/RdmaDevice.cuh"

namespace aboleth::generated {{

namespace rdma = ::aboleth::cuda::rdma;

inline constexpr char k{stem}ProgramSignature[] =
    "{signature}";

struct {struct_name} {{
{fields}
}};

template <
    bool kDeferCompletion,
    std::uint32_t kCompletionInterval = 1>
__device__ __forceinline__ void {function_name}(
    const {struct_name}& bindings,
    std::uint32_t* doorbell_lock) {{
    const auto& endpoint =
        *reinterpret_cast<const ::aboleth::abi::RdmaEndpointDescriptor*>(
            bindings.{binding.endpoint.name});
    constexpr std::uint32_t kCommandCount = {command_count};
    const auto base = rdma::reserve_count(endpoint, kCommandCount);
    const auto end = base + kCommandCount;

{transfer_body}

    bool poll_now = false;
    if constexpr (!kDeferCompletion) {{
        if constexpr (kCompletionInterval <= 1) {{
            poll_now = true;
        }} else {{
            poll_now =
                bindings.{publication.iteration.name} % kCompletionInterval == 0 ||
                bindings.{publication.iteration.name} == bindings.{publication.iterations.name};
        }}
    }}
    const auto epoch = rdma::load_acquire(
        reinterpret_cast<const std::uint64_t*>(
            bindings.{publication.local.pointer.name}));
    rdma::write_atomic_signal(
        endpoint,
        base + {signal_index},
        bindings.{publication.remote.pointer.name},
        bindings.{publication.remote.key.name},
        epoch,
        /*request_completion=*/poll_now);
    rdma::publish(endpoint, doorbell_lock, base, end);
    if (poll_now)
        rdma::poll(endpoint, base, end);
}}

}} // namespace aboleth::generated
"""
    return CudaFragmentArtifact(
        program_name=binding.name,
        signature=signature,
        struct_name=struct_name,
        function_name=function_name,
        command_count=command_count,
        source=source,
    )
