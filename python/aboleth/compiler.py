from __future__ import annotations

import contextvars
import hashlib
import os
import pathlib
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from ._plugin import configure_plugin
from .pattern import CompiledPatternPlan

_PLUGIN_PATH = configure_plugin()

from triton import knobs
from triton._C.libtriton import ir, passes

_BACKEND = contextvars.ContextVar("aboleth_backend", default=None)
_COMPLETION_INTERVAL = contextvars.ContextVar(
    "aboleth_completion_interval",
    default=None,
)
_QUEUE_CAPACITY = contextvars.ContextVar(
    "aboleth_queue_capacity",
    default=None,
)
_GPU_RESIDENT_QUEUES = contextvars.ContextVar(
    "aboleth_gpu_resident_queues",
    default=False,
)
_STRICT_REGISTRATIONS = contextvars.ContextVar(
    "aboleth_strict_registrations",
    default=False,
)
# The endpoint profile (identity and source) the kernel's plan came from.
_ENDPOINT_PROFILE = contextvars.ContextVar(
    "aboleth_endpoint_profile",
    default=None,
)
_PATTERN_PLAN_PATH = contextvars.ContextVar(
    "aboleth_pattern_plan_path",
    default=None,
)
_PATTERN_PLAN_HASH = contextvars.ContextVar(
    "aboleth_pattern_plan_hash",
    default=None,
)
_SUPPORTED_BACKENDS = frozenset(("p2p", "proxy", "rdma", "efa", "ionic"))
_PLAN_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="aboleth-plans-"
)

# Largest payload forwarded from registers into an inline transfer. 28 bytes is
# the largest mlx5 inline payload that still fits one WQEBB.
DEFAULT_INLINE_BYTES = 28
# The wire's own ceiling, mirroring `kMaxInlinePayloadBytes` in
# include/aboleth/Target/RdmaAbi.h: an mlx5 control segment's ds field is six
# bits, so a WQE spans at most 63 16-byte segments. An inline write spends two
# on control and remote address, leaving 61 for the inline-data segment's
# 4-byte header plus the payload: 2 + ceil((4 + bytes) / 16) <= 63 caps the
# payload at 972 bytes. Beyond it the ds field wraps and the WQE is corrupt.
MAX_INLINE_BYTES = 972
# Backends whose lowering realizes an inline-payload command. Forwarding
# register values into one is only legal where the command exists: every other
# lowering leaves `onInline` unset and rejects the rewritten module.
_INLINE_PAYLOAD_BACKENDS = frozenset(("rdma", "p2p", "ionic"))
# Optimizations `ABOLETH_PASSES` may switch, and whether each is on by
# default: "-name" turns one off, "+name" turns one on, comma-separated
# (`ABOLETH_PASSES=-elide-epochs,+defer-announcement`). Every other pass is
# needed to lower at all. `completion-planning` is materialize-completion's
# use of declared reuse; `gpu-doorbell` is the lowering's doorbell scope;
# `invariant-descriptors` and `vector-wqe-stores` are the RDMA lowering's
# descriptor loads and RDMA-write WQE stores; `self-reclaim` is the RDMA
# lowering's queue reclamation for kernels whose own predicated waits defer
# completion (it changes no other kernel).
OPTIMIZATIONS = {
    "shape-specialization": True,
    "endpoint-fanout": True,
    "coalesce-receives": True,
    "valid-extents": True,
    "plan-storage": True,
    "dead-communication": True,
    "completion-planning": True,
    "forward-inline-payloads": True,
    "plan-publication": True,
    "coalesce-submissions": True,
    "order-publications": True,
    "elide-epochs": True,
    "defer-announcement": False,
    "analyze-issuers": True,
    "share-producer-fence": True,
    "ionic-fenced-publication": True,
    "gpu-doorbell": True,
    "invariant-descriptors": True,
    "vector-wqe-stores": False,
    "self-reclaim": True,
}
# Python target policy also changes generated code when the plugin is unchanged.
# Include its revision in Triton's persistent cache identity.
_RDMA_EMISSION_POLICY = "target-defaults-v1"
# Compute capability -> {switch: on}, applied over OPTIMIZATIONS.
TARGET_OPTIMIZATIONS: dict[int, dict[str, bool]] = {}
_INLINE_BYTES = contextvars.ContextVar(
    "aboleth_inline_bytes",
    default=DEFAULT_INLINE_BYTES,
)


def enabled_optimizations(*, capability: int | None = None) -> frozenset[str]:
    enabled = {name for name, default in OPTIMIZATIONS.items() if default}
    # Per-target defaults (none today: every target uses OPTIMIZATIONS).
    # Explicit switches below override them. CPU planning has no target and
    # never queries a GPU merely to choose lowering options. On SM100 the
    # scalar descriptor path measured faster in full-model runs; select it
    # with ABOLETH_PASSES=-invariant-descriptors.
    for name, on in TARGET_OPTIMIZATIONS.get(capability, {}).items():
        (enabled.add if on else enabled.discard)(name)
    for token in os.environ.get("ABOLETH_PASSES", "").split(","):
        token = token.strip()
        if not token:
            continue
        name = token[1:]
        if token[0] not in "+-" or name not in OPTIMIZATIONS:
            raise ValueError(
                f"ABOLETH_PASSES: {token!r} is not +name or -name for a name "
                f"in {sorted(OPTIMIZATIONS)}")
        if token[0] == "+":
            enabled.add(name)
        else:
            enabled.discard(name)
    return frozenset(enabled)


@lru_cache(maxsize=1)
def _plugin_hash() -> str:
    return hashlib.sha256(_PLUGIN_PATH.read_bytes()).hexdigest()


def _efa_bitcode_path() -> str:
    """Device primitives the EFA lowering calls, linked as Triton extern
    bitcode. Built by `cmake --build build`."""
    configured = os.environ.get("ABOLETH_EFA_BITCODE")
    if configured:
        return configured
    architecture = os.environ.get("ABOLETH_EFA_ARCH", "sm_90a")
    name = f"aboleth_efa_{architecture}.bc"
    packaged = pathlib.Path(__file__).resolve().parent / name
    if packaged.is_file():
        return str(packaged)
    root = pathlib.Path(__file__).resolve().parents[2]
    build = pathlib.Path(os.environ.get("ABOLETH_BUILD_DIR", root / "build"))
    return str(build / "lib" / name)


def _rdma_bitcode_path() -> str:
    configured = os.environ.get("ABOLETH_RDMA_BITCODE")
    if configured:
        return configured
    root = pathlib.Path(__file__).resolve().parents[2]
    architecture = os.environ.get("ABOLETH_RDMA_ARCH", "sm_90a")
    return str(
        root
        / "build"
        / "lib"
        / f"aboleth_rdma_{architecture}.bc"
    )


def _rdma_device_library_enabled() -> bool:
    return os.environ.get("ABOLETH_RDMA_DEVICE_LIBRARY", "0").lower() not in (
        "",
        "0",
        "false",
        "off",
    )


def _efa_publication() -> str:
    return os.environ.get("ABOLETH_EFA_PUBLICATION", "fence")


@lru_cache(maxsize=4)
def _efa_bitcode_hash(path: str) -> str:
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        # A missing library is reported by the lowering pass itself; the cache
        # key only has to change when the library does.
        return "missing"


def _pipeline_key() -> tuple[str, str]:
    backend = _BACKEND.get() or "disabled"
    completion_interval = _COMPLETION_INTERVAL.get()
    pattern_plan_hash = _PATTERN_PLAN_HASH.get()
    scalar = os.environ.get(
        "ABOLETH_P2P_SCALAR",
        "",
    ).lower() not in ("", "0", "false", "off")
    cooperative = os.environ.get(
        "ABOLETH_P2P_COOP",
        "1",
    ).lower() not in ("0", "false", "off")
    cooperative_minimum = os.environ.get(
        "ABOLETH_P2P_COOP_MIN",
        "256",
    )
    # Unset: the lowering sizes the chunk from the payload (2048 elements
    # when the size is a run-time value).
    cooperative_elements = os.environ.get(
        "ABOLETH_P2P_COOP_ELEMENTS",
        "auto",
    )
    if backend == "p2p" and scalar:
        copy_mode = "scalar"
    elif backend == "p2p" and cooperative:
        copy_mode = f"cooperative-{cooperative_minimum}-{cooperative_elements}"
    else:
        copy_mode = "wide"
    efa = ""
    if backend == "efa":
        # The generated kernel inlines the device library, so a new library is
        # a new kernel.
        efa = f":{_efa_publication()}:{_efa_bitcode_hash(_efa_bitcode_path())}"
    rdma = ""
    if backend == "rdma":
        # An explicit +invariant-descriptors equals the generic default but
        # overrides SM100's scalar default. The generic enabled set alone
        # therefore cannot distinguish these compilations. Triton's own
        # target key separates architectures; preserve explicit policy too.
        rdma = (
            f":emission-policy-{_RDMA_EMISSION_POLICY}"
            f":targets-{sorted(TARGET_OPTIMIZATIONS.items())}"
            f":explicit-{os.environ.get('ABOLETH_PASSES', '')}"
        )
        if _rdma_device_library_enabled():
            rdma += f":{_efa_bitcode_hash(_rdma_bitcode_path())}"
    inline_bytes = _INLINE_BYTES.get()
    switches = ",".join(sorted(enabled_optimizations()))
    hip = ""
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.version.hip:
        path = str(pathlib.Path(__file__).with_name("aboleth_hip.bc"))
        hip = f":hip-{_efa_bitcode_hash(path)}"
        if backend == "ionic":
            ionic = str(pathlib.Path(__file__).with_name("aboleth_ionic.bc"))
            hip += f":ionic-{_efa_bitcode_hash(ionic)}"
    identity = (
        f"{_plugin_hash()}:{backend}:{copy_mode}"
        f":{completion_interval}:{pattern_plan_hash}"
        f":inline-{inline_bytes}{efa}{rdma}:{switches}"
        f":queue-{_QUEUE_CAPACITY.get()}"
        f":strict-{_STRICT_REGISTRATIONS.get()}"
        f":gpu-queues-{_GPU_RESIDENT_QUEUES.get()}{hip}"
        f":profile-{_ENDPOINT_PROFILE.get()}"
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    # Triton's in-memory cache keys on the digest, but its on-disk cache
    # appends only this string, so the string must carry the whole identity:
    # without the digest, a kernel compiled under one plugin, pass switch, or
    # registration ordering is reused under another.
    key = (
        f"aboleth:{backend}:{copy_mode}:completion-"
        f"{completion_interval}:inline-{inline_bytes}:{digest}"
    )
    return key, digest


@lru_cache(maxsize=32)
def _pattern_plan_path(transport_mlir: str) -> str:
    digest = hashlib.sha256(
        transport_mlir.encode()
    ).hexdigest()
    path = pathlib.Path(_PLAN_DIRECTORY.name) / f"{digest}.mlir"
    path.write_text(transport_mlir, encoding="utf-8")
    return str(path)


def _stages_hook(
    backend=None,
    stages=None,
    options=None,
    language=None,
    capability=None,
):
    if all(value is None for value in (backend, stages, options, language, capability)):
        return _pipeline_key()

    selected = _BACKEND.get()
    if selected is None:
        return _pipeline_key()

    hip_target = getattr(getattr(backend, "target", None), "backend", None) == "hip"
    hip_library = None
    if hip_target:
        if selected not in ("p2p", "proxy", "ionic"):
            raise NotImplementedError("HIP supports p2p, proxy and ionic lowering")
        hip_library = str(pathlib.Path(__file__).with_name("aboleth_hip.bc"))
        if not pathlib.Path(hip_library).is_file():
            raise RuntimeError("HIP device primitives are not built; build aboleth_hip_bitcode")
        if selected == "ionic" and options.num_warps != 1:
            raise NotImplementedError("Ionic kernels currently require num_warps=1")
        libraries = dict(options.extern_libs or ())
        libraries["aboleth_hip"] = hip_library
        if selected == "ionic":
            ionic = str(pathlib.Path(__file__).with_name("aboleth_ionic.bc"))
            if not pathlib.Path(ionic).is_file():
                raise RuntimeError("build aboleth_ionic_bitcode before using Ionic")
            libraries["aboleth_ionic"] = ionic
        object.__setattr__(options, "extern_libs", tuple(libraries.items()))

    if selected == "efa" and options is not None:
        # The EFA lowering emits calls into the device library; Triton links
        # whatever is listed here and inlines it.
        libraries = dict(options.extern_libs or ())
        libraries["aboleth_efa"] = _efa_bitcode_path()
        object.__setattr__(options, "extern_libs", tuple(libraries.items()))
    if selected == "rdma" and _rdma_device_library_enabled() and options is not None:
        libraries = dict(options.extern_libs or ())
        libraries["aboleth_rdma"] = _rdma_bitcode_path()
        object.__setattr__(options, "extern_libs", tuple(libraries.items()))

    stage_name = (
        "ttir"
        if "ttir" in stages
        else "ttgir"
    )
    make_ir = stages[stage_name]

    def lower_aboleth(source, metadata):
        module = make_ir(source, metadata)
        enabled = enabled_optimizations(
            capability=capability if selected == "rdma" else None
        )
        manager = ir.pass_manager(module.context)
        plan_path = _PATTERN_PLAN_PATH.get()
        if plan_path is not None:
            passes.plugin.add_aboleth_materialize_transport_phase(
                manager,
                [plan_path],
            )
        passes.plugin.add_aboleth_materialize_valid_extents(
            manager, ["" if "valid-extents" in enabled else "unoptimized"])
        completion_interval = _COMPLETION_INTERVAL.get()
        queue_capacity = _QUEUE_CAPACITY.get()
        passes.plugin.add_aboleth_materialize_completion(
            manager,
            [
                "" if completion_interval is None else str(completion_interval),
                selected,
                "" if queue_capacity is None else str(queue_capacity),
                "" if "completion-planning" in enabled else "unplanned",
            ],
        )
        # Forwarding runs on semantic puts, before publication strength and
        # batch formation look at them: a forwarded put carries no source
        # region, so every later pass sees the final shape of the transfer.
        # Only where the backend has an inline-payload command: rewriting a
        # put into one the lowering cannot realize turns a working kernel into
        # a hard compilation error.
        if (selected in _INLINE_PAYLOAD_BACKENDS
                and "forward-inline-payloads" in enabled):
            passes.plugin.add_aboleth_forward_inline_payloads(
                manager,
                [str(min(_INLINE_BYTES.get(), 16) if selected == "ionic" else _INLINE_BYTES.get())],
            )
        if "plan-publication" in enabled:
            passes.plugin.add_aboleth_plan_publication(
                manager,
                [selected, "strict" if _STRICT_REGISTRATIONS.get() else ""],
            )
        passes.plugin.add_aboleth_form_batches(manager)
        # Merging precedes producer ordering and issuer analysis, which then
        # see the merged batch: one barrier and one reservation for all of it.
        if "coalesce-submissions" in enabled:
            # Planned completion builds one predicate per wait; folding equal
            # ones lets chains due together share a poll.
            passes.common.add_cse(manager)
            passes.plugin.add_aboleth_coalesce_submissions(manager, [selected])
            if "order-publications" in enabled:
                passes.plugin.add_aboleth_order_publications(
                    manager, [selected])
        # Publish planned loops' epochs only when completion is due (and on
        # the last iteration): receivers wait for "at least e", so a later
        # epoch stands in for the skipped ones.
        if "elide-epochs" in enabled:
            passes.plugin.add_aboleth_elide_epochs(manager, [selected])
        # Ring only when a planned poll is due: faster with few chains per
        # round, slower with many, so opt-in.
        if "defer-announcement" in enabled:
            passes.plugin.add_aboleth_defer_announcement(manager, [selected])
        passes.plugin.add_aboleth_order_producers(manager)
        if "analyze-issuers" in enabled:
            passes.plugin.add_aboleth_analyze_issuers(manager)
        manager.run(module, "aboleth-scheduling")
        if os.environ.get("ABOLETH_DUMP_SCHEDULED_TTIR") == "1":
            print(module.str_nodebug(), flush=True)

        manager = ir.pass_manager(module.context)
        arguments = []
        if selected == "efa":
            arguments = [_efa_publication(), _efa_bitcode_path()]
        elif selected == "rdma":
            arguments = rdma_lowering_options(enabled)
        elif selected == "ionic":
            if hip_library is None:
                raise NotImplementedError("Ionic lowering requires a HIP target")
            arguments = [str(pathlib.Path(__file__).with_name("aboleth_ionic.bc")),
                         "", "share-producer-fence" if "share-producer-fence" in enabled else "",
                         "fenced-publication" if "ionic-fenced-publication" in enabled else ""]
        elif hip_library is not None:
            arguments = ["", "", hip_library] if selected == "p2p" else [hip_library]
        getattr(
            passes.plugin,
            f"add_aboleth_lower_{selected}",
        )(manager, arguments)
        # The lowered code enters after Triton's own TTIR cleanups, so it gets
        # its own: constant ring arithmetic folds and repeated address math
        # merges before layout assignment.
        passes.common.add_canonicalizer(manager)
        passes.common.add_cse(manager)
        # TODO: endpoint-descriptor loads are still emitted per use;
        # cache them per endpoint so they hoist out of loops.
        manager.run(module, f"aboleth-lower-{selected}")
        return module

    stages[stage_name] = lower_aboleth
    return _pipeline_key()


def rdma_lowering_options(enabled: frozenset[str], *,
                          detached: bool = False,
                          exclusive: bool = False,
                          gpu_resident_queues: bool | None = None) -> list[str]:
    """The RDMA lowering's named options (`name=value`) for this compilation.

    `gpu_resident_queues` defaults to the enclosing `compilation()` scope's
    placement; a detached fragment has no endpoint, so its caller states it
    (the Tier-1 library builder, from the World it is built for).
    """
    if _rdma_device_library_enabled():
        options = [f"bitcode-path={_rdma_bitcode_path()}"]
    else:
        options = [name for name in ("invariant-descriptors",
                                     "vector-wqe-stores") if name in enabled]
    if detached:
        options.append("detached")
    if exclusive:
        options.append("exclusive")
    # Reclamation lowers only in-kernel modules that ask for it (see
    # LowerRDMA); detached fragments never do.
    if (not detached and "self-reclaim" in enabled
            and not _rdma_device_library_enabled()):
        options.append("self-reclaim")
    # Every queue is in GPU memory, and so is every payload source
    # (registration takes only CUDA tensors): a GPU-scope release publishes
    # the doorbell.
    if gpu_resident_queues is None:
        gpu_resident_queues = not detached and _GPU_RESIDENT_QUEUES.get()
    if (gpu_resident_queues
            and "gpu-doorbell" in enabled
            and not _rdma_device_library_enabled()):
        options.append("gpu-scope-doorbell")
    return options


def install_compiler() -> None:
    knobs.runtime.add_stages_inspection_hook = _stages_hook


@contextmanager
def compilation(
    backend: str,
    *,
    completion_interval: int | None = None,
    pattern_plan: CompiledPatternPlan | None = None,
    kernel: str | None = None,
    inline_bytes: int = DEFAULT_INLINE_BYTES,
    p2p_coop_elements: int | None = None,
    queue_capacity: int | None = None,
    strict_registrations: bool = False,
    gpu_resident_queues: bool = False,
    endpoint_profile: str | None = None,
) -> Iterator[None]:
    """Compile one Aboleth communication phase for one backend.

    `queue_capacity` is the smallest send-queue capacity, in slots, of the
    endpoints the kernels use; completion planning needs it to bound how long
    a kernel that declares buffer reuse may leave submissions outstanding.

    `strict_registrations` states that every registration the kernels publish
    into is strictly ordered (no relaxed ordering). Only then may a strong
    signal be a fenced write rather than a responder atomic.

    `gpu_resident_queues` states that every endpoint the kernels post to keeps
    its queues in GPU memory; the doorbell then needs only a GPU-scope release
    instead of a system-scope one.

    `endpoint_profile` names the endpoint profile (World.endpoint_profiles
    identity) that planned the kernels' lanes, so a kernel compiled under
    one profile is never reused under another.
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported Aboleth backend: {backend}")
    if p2p_coop_elements is not None:
        p2p_coop_elements = int(p2p_coop_elements)
        if (
            p2p_coop_elements < 32
            or p2p_coop_elements > 4096
            or p2p_coop_elements & (p2p_coop_elements - 1)
        ):
            raise ValueError(
                "p2p_coop_elements must be a power of two in [32, 4096]"
            )
    inline_bytes = int(inline_bytes)
    if inline_bytes < 0:
        raise ValueError("inline_bytes cannot be negative")
    if inline_bytes > MAX_INLINE_BYTES:
        raise ValueError(
            f"inline_bytes={inline_bytes} exceeds the {MAX_INLINE_BYTES}-byte "
            "wire maximum for an inline payload (an mlx5 control segment's "
            "6-bit ds field spans at most 63 16-byte segments)"
        )
    if completion_interval is not None:
        completion_interval = int(completion_interval)
        if completion_interval <= 0:
            raise ValueError("completion interval must be positive")
    if pattern_plan is not None:
        if pattern_plan.transport_plan and not kernel:
            raise ValueError("kernel is required when applying a transport plan")
        planned_interval = (
            pattern_plan.completion_interval_for(kernel) if kernel else None
        )
        if completion_interval is None:
            completion_interval = planned_interval
        elif planned_interval is not None and completion_interval != planned_interval:
            raise ValueError("completion_interval disagrees with pattern_plan")
    plan_path = None
    if pattern_plan is not None:
        plan_path = _pattern_plan_path(
            pattern_plan.transport_mlir
        )
        plan_hash = hashlib.sha256(pattern_plan.transport_mlir.encode()).hexdigest()
    else:
        plan_hash = None
    backend_token = _BACKEND.set(backend)
    completion_token = _COMPLETION_INTERVAL.set(completion_interval)
    plan_path_token = _PATTERN_PLAN_PATH.set(plan_path)
    plan_hash_token = _PATTERN_PLAN_HASH.set(plan_hash)
    inline_token = _INLINE_BYTES.set(inline_bytes)
    capacity_token = _QUEUE_CAPACITY.set(
        None if queue_capacity is None else int(queue_capacity)
    )
    strict_token = _STRICT_REGISTRATIONS.set(bool(strict_registrations))
    queues_token = _GPU_RESIDENT_QUEUES.set(bool(gpu_resident_queues))
    profile_token = _ENDPOINT_PROFILE.set(endpoint_profile)
    # The lowering reads the chunk width from the environment. A planner
    # choice is installed only for this compilation scope, and an explicit
    # environment setting stays the operator's override.
    coop_installed = (
        p2p_coop_elements is not None
        and "ABOLETH_P2P_COOP_ELEMENTS" not in os.environ
    )
    if coop_installed:
        os.environ["ABOLETH_P2P_COOP_ELEMENTS"] = str(p2p_coop_elements)
    try:
        yield
    finally:
        if coop_installed:
            os.environ.pop("ABOLETH_P2P_COOP_ELEMENTS", None)
        _ENDPOINT_PROFILE.reset(profile_token)
        _GPU_RESIDENT_QUEUES.reset(queues_token)
        _STRICT_REGISTRATIONS.reset(strict_token)
        _QUEUE_CAPACITY.reset(capacity_token)
        _INLINE_BYTES.reset(inline_token)
        _PATTERN_PLAN_HASH.reset(plan_hash_token)
        _PATTERN_PLAN_PATH.reset(plan_path_token)
        _COMPLETION_INTERVAL.reset(completion_token)
        _BACKEND.reset(backend_token)


install_compiler()
