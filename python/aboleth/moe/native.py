"""Executor for a compiler-planned BF16 MoE program.

This owns CUDA allocations and executes the compiled launch list. It does not
choose capacities, layouts, lanes, transport partitions or copy elimination.
A slot is ordered by its client stream; independent in-flight slots have
separate payload, flags, epoch and endpoint reservations.
"""

from dataclasses import asdict
from functools import partial
from bisect import bisect_left
import torch
import triton

from ..records import send_records, await_records
from .compiled_plan import CompiledMoEPlan
from .kernels import (
    pack_send_records,
    pack_records,
    unpack_records,
    unpack_records_fp8,
    gather_partials,
)

DTYPES = {
    "bf16": torch.bfloat16,
    "f8E4M3FN": torch.float8_e4m3fn,
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
}


class NativeMoESlot:
    # The fused programs' warps and widest row chunk: one constant, the best
    # FP8 tile and within 5% of the best BF16 one (the fused-pack driver's
    # fused:WxC arms override it per slot).
    fused_tile = (8, 4096)

    def __init__(
        self,
        world,
        plan: CompiledMoEPlan,
        *,
        owners: torch.Tensor,
        placeholder: int,
        tag: str,
        endpoint_base: int,
        endpoint_capacity: int,
        fuse_pack: bool = False,
    ):
        self.world, self.plan = world, plan
        # Pack each peer's records inside its dispatch program, before the
        # puts, instead of in a separate kernel ahead of the transport.
        self.fuse_pack = fuse_pack
        self.spec = s = plan.spec
        self.device, self.rank = world.device, world.rank
        if (
            owners.shape != (s.experts,)
            or owners.dtype != torch.int32
            or not owners.is_cuda
        ):
            raise ValueError("expert owners must be a CUDA int32 vector")
        if (
            not 0 <= placeholder < s.experts
            or int(owners[placeholder].item()) == self.rank
        ):
            raise ValueError(
                "inactive expert placeholder must belong to another rank"
            )
        if (
            endpoint_base < 0
            or endpoint_capacity < 1
            or endpoint_base + endpoint_capacity > world.endpoints_per_peer
        ):
            raise ValueError(
                "slot endpoint reservation exceeds connected resources"
            )
        self.placeholder = placeholder
        requirements = {r.name: r for r in plan.pattern.regions}
        self.input_bindings = tuple(
            (
                name,
                DTYPES[requirements[name].element_type],
                plan.pattern.storage_plan[name].prefix_binding,
            )
            for name in ("tokens", "input_ids", "input_weights")
        )
        self.buffers = {
            name: torch.empty(
                storage.physical_shape,
                device=self.device,
                dtype=DTYPES[requirements[name].element_type],
            )
            for name, storage in plan.pattern.storage_plan.items()
            if storage.prefix_binding is None
        }
        self._owned_buffers = dict(self.buffers)
        self.buffers["owners"].copy_(owners)
        for tensor in self.buffers.values():
            if tensor.dtype.is_floating_point:
                tensor.fill_(float("nan"))
            else:
                tensor.fill_(-12345)
        self.buffers["owners"].copy_(owners)
        self.epoch = torch.ones(1, device=self.device, dtype=torch.int64)
        self.flags = {
            phase: torch.zeros(
                launches[0].signal_slots, device=self.device, dtype=torch.int64
            )
            for phase, launches in [
                ("dispatch", plan.dispatch_launches),
                ("return", plan.return_launches),
            ]
        }
        forbidden = sorted(
            (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
            for t in (
                *self._owned_buffers.values(),
                self.epoch,
                *self.flags.values(),
            )
        )
        self._input_starts = tuple(start for start, _ in forbidden)
        self._input_ends = tuple(end for _, end in forbidden)
        # Peer-memory-only programs need IPC exports, not NIC memory regions.
        # Retain network registration for every mixed or network geography.
        register = partial(
            world.runtime.register,
            network=any(kind != "p2p" for _, kind in world.transport_geography),
        )
        registered = {
            name: world.region(
                name, register(self.buffers[name]), self.buffers[name]
            )
            for name, r in requirements.items()
            if r.registered and name in self.buffers
        }
        registered.update(
            {
                phase
                + "_flags": world.region(
                    phase + "_flags", register(flags), flags
                )
                for phase, flags in self.flags.items()
            }
        )
        self.table = world.exchange_regions(registered, tag=tag)
        self.launches = {}
        self.compiled_kernels = {}
        for phase, launches in [
            ("dispatch", plan.dispatch_launches),
            ("return", plan.return_launches),
        ]:
            selected = []
            for launch in launches:
                if launch.lanes > endpoint_capacity:
                    raise ValueError("compiled lanes exceed slot reservation")
                # uint64 advanced indexing is unsupported by torch. This is
                # a bit-preserving int64 view, never a numerical handle cast.
                handles = (
                    self.table.endpoint_handles.view(torch.int64)
                    .view(s.ranks, world.endpoints_per_peer)[
                        list(launch.peers),
                        endpoint_base : endpoint_base + launch.lanes,
                    ]
                    .T.contiguous()
                    .flatten()
                    .view(torch.uint64)
                )
                selected.append((launch, handles))
            self.launches[phase] = tuple(selected)
        self.calls = {
            phase: self._bind_transport(phase, endpoint_base)
            for phase in self.launches
        }
        self._actual_tokens = 0

    @property
    def expert_input(self):
        return self.buffers["received_payload"][
            : self.spec.ranks * self.spec.capacity
        ]

    @property
    def expert_output(self):
        return self.buffers[self.plan.expert_destination]

    @property
    def expert_scales(self):
        return self.buffers.get("expert_scales")

    def _bind_transport(self, phase, endpoint_base):
        s = self.spec
        t = self.table
        b = self.buffers
        fields = (
            self.plan.dispatch_fields
            if phase == "dispatch"
            else self.plan.return_fields
        )
        source = "send_payload" if phase == "dispatch" else "return_source"
        meta = "send_metadata" if phase == "dispatch" else "received_metadata"
        counts = "send_counts" if phase == "dispatch" else "received_counts"
        destination = (
            "received_payload" if phase == "dispatch" else "returned_payload"
        )
        destination_meta = (
            "received_metadata" if phase == "dispatch" else "returned_metadata"
        )
        destination_counts = (
            "received_counts" if phase == "dispatch" else "returned_counts"
        )
        # A removed field has no allocation, descriptor or command. Dummy
        # arguments are folded out by the semantic kernel's constexpr branch.
        metadata = b[meta] if "metadata" in fields else b[source]
        metadata_key = (
            t.local(meta).lkey
            if "metadata" in fields
            else t.local(source).lkey
        )
        remote_meta = (
            t.regions(destination_meta)
            if "metadata" in fields
            else t.regions(destination)
        )
        remote_counts = (
            t.regions(destination_counts)
            if "count" in fields
            else t.regions(destination)
        )
        phase_name = (
            self.plan.dispatch_phase
            if phase == "dispatch"
            else self.plan.return_phase
        )
        completion = self.plan.pattern.transport_plan[phase_name]
        reuse, interval = (
            completion.reuse_distance,
            completion.completion_interval,
        )
        calls = []
        for launch, handles in self.launches[phase]:
            endpoints = tuple(
                self.world.endpoint(peer, slot=endpoint_base + lane)
                for peer in launch.peers
                for lane in range(launch.lanes)
            )
            scope = partial(
                endpoints[0].compilation,
                completion_interval=interval,
                queue_capacity=min(
                    endpoint.capacity for endpoint in endpoints
                ),
                strict_registrations=t.strictly_ordered,
                gpu_resident_queues=all(
                    endpoint.gpu_resident_queues for endpoint in endpoints
                ),
                endpoint_profile=launch.profile_identity,
            )
            arguments = (
                handles,
                b[source],
                t.local(source).lkey,
                metadata,
                metadata_key,
                t.regions(destination),
                remote_meta,
                remote_counts,
                t.regions(phase + "_flags"),
                b[counts],
                self.epoch,
            )
            constants = dict(
                RANK=self.rank,
                PEERS=launch.peers,
                CAPACITY=s.capacity,
                WIDTH=s.hidden,
                META_WIDTH=s.metadata_words if "metadata" in fields else 0,
                LANES=launch.lanes,
                ROW_QUOTIENT=launch.slice_quotient,
                ROW_REMAINDER=launch.slice_remainder,
                REUSE=reuse,
                READY_STRIDE=launch.ready_stride,
                COUNT_PRESENT="count" in fields,
                num_warps=launch.sender_warps,
            )
            calls.append((launch, arguments, constants, scope))
        return tuple(calls)

    def _pack_constants(self, num_warps, columns=1024):
        s = self.spec
        # A [BLOCK_T, BLOCK_C] tile of rows per step, about 16 elements per
        # thread; quantization groups never straddle a chunk.
        chunk = max(min(triton.next_power_of_2(s.hidden), columns), s.scale_group)
        return dict(
            BLOCK_T=max(2, min(16, 512 * num_warps // chunk)),
            BLOCK_C=chunk,
            QUANT_GROUP=s.scale_group,
            ROUND_SCALE=s.round_scale,
            AMAX_FLOOR=s.quant_amax_floor,
        )

    def _pack_inputs(self, n):
        b = self.buffers
        if "tokens" in b:
            return b["tokens"], b["input_ids"], b["input_weights"], n
        # Before the first dispatch a borrowed input has no binding yet; the
        # inactive warmup launch compiles against same-typed placeholders.
        return (
            torch.empty(1, device=self.device, dtype=torch.bfloat16),
            torch.empty(1, device=self.device, dtype=torch.int32),
            torch.empty(1, device=self.device, dtype=torch.float32),
            0,
        )

    def _transport(self, phase, *, active=1, num_tokens=0):
        # Registration and compiler facts are invariant for this artifact.
        # Construct its contexts/descriptors once, rather than scanning every
        # variant's endpoint pool during each eager MoE invocation.
        for call in self.calls[phase]:
            self._send(phase, call, active, num_tokens)
        self._await(phase, active)

    def _send(self, phase, call, active, num_tokens):
        launch, arguments, constants, scope = call
        with scope():
            if phase == "dispatch" and self.fuse_pack:
                kernel = self._pack_send(
                    launch, arguments, constants, active, num_tokens
                )
            else:
                kernel = send_records[launch.sender_grid](
                    *arguments, active, **constants
                )
        self.compiled_kernels[(phase, launch.backend)] = kernel

    def _await(self, phase, active):
        for launch, _, _, scope in self.calls[phase]:
            with scope():
                await_records[launch.receiver_grid](
                    self.flags[phase],
                    self.epoch,
                    active,
                    PEERS=launch.peers,
                    LANES=launch.lanes,
                    READY_STRIDE=launch.ready_stride,
                    LANES_PER_PROGRAM=launch.receiver_lanes_per_program,
                    num_warps=launch.receiver_warps,
                )

    def _pack_send(self, launch, arguments, constants, active, num_tokens):
        s = self.spec
        b = self.buffers
        (
            handles,
            payload,
            payload_key,
            metadata,
            metadata_key,
            data,
            remote_meta,
            remote_counts,
            flags,
            _,
            epoch,
        ) = arguments
        tokens, ids, weights, num_tokens = self._pack_inputs(
            num_tokens if active else 0
        )
        warps, columns = self.fused_tile
        if launch.backend != "ionic":
            warps = max(constants["num_warps"], warps)
        else:
            warps = constants["num_warps"]
        return pack_send_records[launch.sender_grid](
            handles,
            tokens,
            ids,
            weights,
            b["owners"],
            payload,
            payload_key,
            metadata,
            metadata_key,
            b["source_slots"],
            b["overflow"],
            data,
            remote_meta,
            remote_counts,
            flags,
            epoch,
            num_tokens,
            active,
            RANK=self.rank,
            PEERS=launch.peers,
            CAPACITY=s.capacity,
            HIDDEN=s.hidden,
            TOP_K=s.top_k,
            NUM_EXPERTS=s.experts,
            WORLD_SIZE=s.ranks,
            METADATA_WORDS=s.metadata_words,
            LANES=launch.lanes,
            ROW_QUOTIENT=launch.slice_quotient,
            ROW_REMAINDER=launch.slice_remainder,
            REUSE=constants["REUSE"],
            READY_STRIDE=launch.ready_stride,
            num_warps=warps,
            **self._pack_constants(warps, columns),
        )

    def warmup_transport(self):
        self._transport("dispatch", active=0)
        self._transport("return", active=0)

    @staticmethod
    def bind_input_domain(handles):
        """Exclude owned storage across every variant and concurrent slot."""
        intervals = sorted(
            (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
            for handle in handles
            for t in (
                *handle._owned_buffers.values(),
                handle.epoch,
                *handle.flags.values(),
            )
        )
        starts = tuple(start for start, _ in intervals)
        ends = tuple(end for _, end in intervals)
        for handle in handles:
            handle._input_starts, handle._input_ends = starts, ends

    def dispatch(self, tokens, ids, weights):
        s = self.spec
        b = self.buffers
        n = int(tokens.shape[0])
        if (
            tokens.shape != (n, s.hidden)
            or tokens.dtype != torch.bfloat16
            or n > s.capacity
        ):
            raise ValueError("input must be bounded [tokens,hidden] BF16")
        if ids.shape != (n, s.top_k) or weights.shape != ids.shape:
            raise ValueError("route shapes must match the declared top-k")
        if any(
            not value.is_cuda or value.device.index != self.device
            for value in (tokens, ids, weights)
        ):
            raise ValueError("MoE inputs must use the artifact CUDA device")
        self._actual_tokens = n
        for (name, dtype, permission), value in zip(
            self.input_bindings, (tokens, ids, weights), strict=True
        ):
            if permission is not None:
                if (
                    permission["count"] != "num_tokens"
                    or n > permission["capacity"]
                ):
                    raise ValueError(
                        "input does not satisfy its compiled prefix binding"
                    )
                bound = value.to(dtype).contiguous()
                length = bound.numel() * bound.element_size()
                if length != n * permission["row_bytes"]:
                    raise ValueError(
                        "input layout violates the compiled row bound"
                    )
                if length:
                    begin = bound.data_ptr()
                    index = bisect_left(self._input_starts, begin + length)
                    if index and self._input_ends[index - 1] > begin:
                        raise ValueError(
                            "borrowed input overlaps artifact storage"
                        )
                b[name] = bound
            else:
                b[name][:n].copy_(value)
        b["overflow"].zero_()
        b["expert_counts"].zero_()
        if self.fuse_pack:
            # The send programs pack the peers' records; the rank's own
            # records go straight into its receive storage, one program per
            # token as in the separate pack.
            b["received_counts"][self.rank : self.rank + 1].zero_()
            if n:
                pack_records[(n, 1)](
                    b["tokens"],
                    b["input_ids"],
                    b["input_weights"],
                    b["owners"],
                    b["received_payload"],
                    b["received_metadata"],
                    b["received_counts"],
                    b["source_slots"],
                    b["overflow"],
                    n,
                    HIDDEN=s.hidden,
                    TOP_K=s.top_k,
                    NUM_EXPERTS=s.experts,
                    WORLD_SIZE=s.ranks,
                    MAX_TOKENS=s.capacity,
                    METADATA_WORDS=s.metadata_words,
                    BLOCK_H=triton.next_power_of_2(s.hidden),
                    QUANT_GROUP=s.scale_group,
                    ROUND_SCALE=s.round_scale,
                    AMAX_FLOOR=s.quant_amax_floor,
                    DESTINATION_BASE=self.rank,
                    num_warps=4,
                )
            self._transport("dispatch", num_tokens=n)
            return self._unpack()
        b["send_counts"].zero_()
        # Match the accepted algorithm's actual-row workset. Capacity bounds
        # transport and allocation; it does not require dead packing CTAs.
        if n:
            pack_records[(n, s.ranks)](
                b["tokens"],
                b["input_ids"],
                b["input_weights"],
                b["owners"],
                b["send_payload"],
                b["send_metadata"],
                b["send_counts"],
                b["source_slots"],
                b["overflow"],
                n,
                HIDDEN=s.hidden,
                TOP_K=s.top_k,
                NUM_EXPERTS=s.experts,
                WORLD_SIZE=s.ranks,
                MAX_TOKENS=s.capacity,
                METADATA_WORDS=s.metadata_words,
                BLOCK_H=triton.next_power_of_2(s.hidden),
                QUANT_GROUP=s.scale_group,
                ROUND_SCALE=s.round_scale,
                AMAX_FLOOR=s.quant_amax_floor,
                num_warps=4,
            )
        begin = self.rank * s.capacity
        end = begin + s.capacity
        b["received_payload"][begin:end].copy_(b["send_payload"][begin:end])
        b["received_metadata"][begin:end].copy_(b["send_metadata"][begin:end])
        b["received_counts"][self.rank : self.rank + 1].copy_(
            b["send_counts"][self.rank : self.rank + 1]
        )
        self._transport("dispatch")
        return self._unpack()

    def _unpack(self):
        s = self.spec
        b = self.buffers
        args = (
            b["received_metadata"],
            b["received_counts"],
            b["expert_counts"],
            b["expert_ids"],
            b["expert_weights"],
            b["overflow"],
        )
        constants = dict(
            TOP_K=s.top_k,
            WORLD_SIZE=s.ranks,
            MAX_TOKENS=s.capacity,
            METADATA_WORDS=s.metadata_words,
            NUM_EXPERTS=s.experts,
            PLACEHOLDER=self.placeholder,
            num_warps=1,
        )
        if s.scale_cols:
            unpack_records_fp8[(s.ranks * s.capacity,)](
                *args,
                b["expert_scales"],
                SCALE_COLS=s.scale_cols,
                BLOCK_SCALES=triton.next_power_of_2(s.scale_cols),
                **constants,
            )
        else:
            unpack_records[(s.ranks * s.capacity,)](*args, **constants)
        return (
            self.expert_input,
            b["expert_counts"],
            b["expert_ids"],
            b["expert_weights"],
        )

    def combine(self, expert_output):
        s = self.spec
        b = self.buffers
        if (
            expert_output.shape != (s.ranks * s.capacity, s.hidden)
            or expert_output.dtype != torch.bfloat16
        ):
            raise ValueError("expert output violates the declared interface")
        if expert_output.data_ptr() != self.expert_output.data_ptr():
            self.expert_output.copy_(expert_output)
        if self.plan.residual_expert_copy:
            b["return_source"].copy_(self.expert_output)
        self._transport("return")
        if self._actual_tokens:
            gather_partials[(self._actual_tokens,)](
                b["returned_payload"],
                b["return_source"],
                b["source_slots"],
                b["final_output"],
                b["overflow"],
                self._actual_tokens,
                HIDDEN=s.hidden,
                WORLD_SIZE=s.ranks,
                MAX_TOKENS=s.capacity,
                BLOCK_H=triton.next_power_of_2(s.hidden),
                LOCAL_RANK=self.rank,
                num_warps=4,
            )
        self.epoch.add_(1)
        return b["final_output"][: self._actual_tokens]

    def manifest(self):
        profiles = self.world.resolved_endpoint_profiles
        return dict(
            spec=asdict(self.spec),
            endpoint_profiles=None if profiles is None else profiles.record(),
            dispatch_fields=self.plan.dispatch_fields,
            return_fields=self.plan.return_fields,
            expert_destination=self.plan.expert_destination,
            residual_expert_copy=self.plan.residual_expert_copy,
            storage={
                name: asdict(storage)
                for name, storage in self.plan.pattern.storage_plan.items()
            },
            allocated_root_bytes=sum(
                t.numel() * t.element_size()
                for t in self._owned_buffers.values()
            ),
            borrowed_inputs=tuple(
                name
                for name, storage in self.plan.pattern.storage_plan.items()
                if storage.prefix_binding is not None
            ),
            launches={
                phase: [asdict(plan) for plan, _ in launches]
                for phase, launches in self.launches.items()
            },
            natural_registers={
                phase + "/" + backend: kernel.n_regs
                for (phase, backend), kernel in self.compiled_kernels.items()
            },
            natural_warps={
                phase
                + "/"
                + backend: getattr(kernel.metadata, "num_warps", None)
                for (phase, backend), kernel in self.compiled_kernels.items()
            },
            natural_spills={
                phase + "/" + backend: getattr(kernel, "n_spills", None)
                for (phase, backend), kernel in self.compiled_kernels.items()
            },
        )

    def close(self):
        self.table.close()


class NativeMoEFamily:
    """Instantiate the C++ shape family and its private control domains."""

    RESOURCE_LANES = (
        16  # Qualified platform pool budget, not a per-workload choice.
    )

    def __init__(
        self,
        *,
        rank,
        ranks,
        device,
        bootstrap,
        capacity,
        hidden,
        top_k,
        experts,
        slots,
        owners,
        internode,
        tag,
        iteration_bound=None,
        value_type="bf16",
        scale_group=0,
        round_scale=False,
        quant_amax_floor=1e-4,
        fuse_pack=False,
        remote_transport="rdma",
        hca=None,
    ):
        from ..world import World
        from ..runtime import Transport

        remote_transport = Transport(remote_transport)
        if remote_transport not in (Transport.RDMA, Transport.PROXY, Transport.IONIC):
            raise ValueError("remote transport must be rdma, proxy or ionic")
        from ..shapes import plan_shape_family
        from .compiled_plan import MoESpec, compile_moe_plan

        MoESpec(
            capacity,
            hidden,
            top_k,
            experts,
            ranks,
            value_type,
            scale_group,
            round_scale,
            quant_amax_floor,
        )
        self.family = plan_shape_family(
            name=tag,
            minimum_bound=1,
            maximum_bound=capacity,
            slots=slots,
            uniform_selector=True,
            iteration_bound=iteration_bound,
        )
        controls = max(
            v.control_base + v.control_count for v in self.family.variants
        )
        self.world = world = World(
            rank, ranks, device=device, bootstrap=bootstrap, hca=hca
        )
        world.connect_all(
            "auto" if internode else "p2p",
            remote_transport=remote_transport,
            endpoints_per_peer=controls * self.RESOURCE_LANES,
        )
        registration = world.endpoint_registration(self.RESOURCE_LANES)
        # Measured once per platform and cached; every rank plans with the
        # same profiles (or none, which is one lane).
        self.endpoint_profiles = world.endpoint_profiles()
        self.registration_facts = registration.plan_facts()
        capacities = {
            kind: min(
                world.endpoint(peer).capacity
                for peer, backend in world.transport_geography
                if backend == kind
            )
            for kind in set(
                backend for _, backend in world.transport_geography
            )
        }
        owner_tensor = torch.tensor(owners, device=device, dtype=torch.int32)
        placeholder = next(
            expert for expert, owner in enumerate(owners) if owner != rank
        )
        self.local_experts = tuple(
            expert for expert, owner in enumerate(owners) if owner == rank
        )
        self.local_indices = torch.tensor(
            self.local_experts, device=device, dtype=torch.int64
        )
        self.variants = {}
        for variant in self.family.variants:
            plan = compile_moe_plan(
                MoESpec(
                    variant.capacity,
                    hidden,
                    top_k,
                    experts,
                    ranks,
                    value_type,
                    scale_group,
                    round_scale,
                    quant_amax_floor,
                ),
                geography=world.transport_geography,
                capacities=capacities,
                available_lanes=self.RESOURCE_LANES,
                profiles=self.endpoint_profiles.profiles,
                **self.registration_facts,
            )
            handles = []
            for slot in range(slots):
                handle = NativeMoESlot(
                    world,
                    plan,
                    owners=owner_tensor,
                    placeholder=placeholder,
                    tag=f"{tag}.{variant.capacity}.{slot}",
                    endpoint_base=variant.control_domain(slot)
                    * self.RESOURCE_LANES,
                    endpoint_capacity=self.RESOURCE_LANES,
                    fuse_pack=fuse_pack,
                )
                handle.warmup_transport()
                handles.append(handle)
            self.variants[variant.capacity] = tuple(handles)
        NativeMoESlot.bind_input_domain(
            tuple(
                handle
                for handles in self.variants.values()
                for handle in handles
            )
        )
        self.capacity = capacity
        self.ranks = ranks
        self.hidden = hidden
        self.experts = experts
        self.slots = slots

    def select(self, uniform_bound, slot):
        variant = self.family.select(uniform_bound)
        if not 0 <= slot < self.slots:
            raise ValueError("in-flight slot exceeds the declared family")
        return self.variants[variant.capacity][slot]

    def local_counts(self, global_counts):
        begin = self.local_experts[0]
        end = begin + len(self.local_experts)
        if self.local_experts == tuple(range(begin, end)):
            return global_counts[begin:end]
        return global_counts.index_select(0, self.local_indices)

    def close(self):
        for handles in self.variants.values():
            for handle in handles:
                handle.close()
        self.world.close()
