# Aboleth World

`World` is the rank-level ownership layer above `Runtime`.

It owns:

- One CUDA device and topology-selected HCA
- One transport-selected local endpoint per remote rank
- Topology and connection metadata exchange
- Registered local regions and provider-neutral remote allocation handles
- GPU tensors containing endpoint handles, remote addresses, and rkeys

## Basic flow

```python
from aboleth import World

estimated_peer_payload = 16 * 1024
with World(
    rank,
    world_size,
    device=rank,
    bootstrap=bootstrap,
) as world:
    world.connect_all(
        "auto",
        payload_bytes=estimated_peer_payload,
    )

    registration = world.register(arena)
    regions = {
        "tokens": world.region(
            "tokens",
            registration,
            token_view,
        ),
        "signals": world.region(
            "signals",
            registration,
            signal_view,
        ),
    }
    peers = world.exchange_regions(
        regions,
        tag="dispatch",
    )

    endpoint_handles = peers.endpoint_handles
    remote_token_addresses = peers.addresses("tokens")
    remote_token_rkeys = peers.rkeys("tokens")
```

Named regions may be complete registrations or views inside one registered
arena. Region exchange preserves offsets and lengths. Same-node peers receive
CUDA IPC handles and importer-local mapped addresses; remote verbs peers
receive exporter addresses and rkeys.

`connect_all("auto", payload_bytes=...)` gathers node/device topology and
applies `AutoTransportPolicy`. On Linux, the advertised node identity combines
the hostname with the kernel boot UUID from `/proc/sys/kernel/random/boot_id`.
Local ranks share it, while separate machines cloned with the same hostname
remain remote peers. The existing hostname string field carries this identity;
the wire framing is unchanged. Platforms without a readable, valid boot UUID
retain hostname-based identification.

Its calibrated default chooses P2P for
same-node peer-accessible GPUs through a 16 KiB estimated peer payload and the
configured RDMA or proxy transport above that threshold or for remote peers.
The immutable policy object is replaceable; omitting `payload_bytes` selects
from topology alone. A world may therefore contain P2P and verbs partitions.

`connect_all(..., endpoints_per_peer=N)` (default 1) creates N endpoints per
peer for multi-CTA device code:
`PeerRegionTable.endpoint_handles` becomes peer-major `[world_size × N]`, a
Tier-1 library generated for the same N selects slot `blockIdx.x % N`, and
each CTA then owns its endpoint — `quiet` stays single-consumer per endpoint
without any locking, so multi-CTA RDMA traffic is not serialized on one
queue. `World.endpoint(peer,
slot=...)` addresses individual slots; `world.endpoints` and
`world.endpoint(peer)` remain the slot-0 view every single-endpoint consumer
expects. Ordering across slots to one peer is unordered; per slot it stays
FIFO. Cost is N queue pairs per peer, so fan-out is opt-in.

CUDA IPC mappings are cached by allocation handle and offset, reference
counted across named regions, generation checked on import, and closed before
the exporter releases its arena. `AllocationProvider.CUDA_VMM` is reserved
behind the same handle interface for a future VMM implementation.

Call `PeerRegionTable.close()` collectively before reusing an exchange tag or
re-registering its exported arenas. Workload drivers close imported mappings,
synchronize ranks, and only then allow exporter registrations to be released.

`World.transport_geography` exposes the canonical connected peer-rank to
transport mapping used by every compiler tier.
`world.device_library(specialize=True)` bakes it into Tier 1. Higher-tier
dynamic routing uses
`world.compilation(peers=...)` on backend-homogeneous partitions; a mixed
partition is rejected instead of being lowered through a representative
backend or a runtime transport-tag branch.

## Endpoint facts and profiles

`World.endpoint_registration(lanes)` gathers every rank's platform key (GPU,
NIC vendor, device and firmware, link rate, submission mode), queue memory,
endpoint budget and local peer link: the canonical image `plan-endpoints`
matches profiles against. `World.endpoint_profiles(pin=(), calibrate=True)`
resolves the profiles a world plans fan-out with, measuring them on first
contact with a platform (`docs/COMPILER_SEMANTICS.md`, endpoint planning).
Both are collective.

## Bootstrap

The `Bootstrap` protocol has one operation:

```python
all_gather(tag: str, payload: bytes) -> dict[int, bytes]
```

World uses it for versioned connection and region bundles. Two
implementations ship: `PipeBootstrap` (same-node, coordinated by
`PipeBootstrapCoordinator`) and `TcpStoreBootstrap` (multi-node over a
`torch.distributed.TCPStore`, rendezvoused by MASTER_ADDR/MASTER_PORT/RANK/
WORLD_SIZE and launched by `scripts/run-multinode.sh` under SLURM). An MPI or
framework-specific bootstrap can implement the same protocol without changing
World or compiler-facing tables.

## Current limits

- CUDA IPC works only within one physical host and requires peer-accessible
  GPU pairs. VMM shareable handles are not implemented yet.
- `connect_all()` still establishes every peer edge; manifest-derived sparse
  connection plans and bounded QP pools remain future work.
- Automatic size selection currently chooses one endpoint provider per peer at
  connection time. Keeping both providers live and selecting independently for
  each compiled phase remains future work.
- Tier-2 kernels bind regions as objects (`aboleth.region(ptr, key)`,
  `aboleth.remote(table.regions(name), peer)`), but keys still travel as
  kernel arguments; frontend-neutral resource binding would hide them.
