# Aboleth transport control

What a caller may decide about the transport under a compiled program, and
what it may not decide yet.

Aboleth's control surface is small on purpose. A queue pair is a physical
resource whose shape the compiler plans against — queue capacity bounds a
deferral window, granted inline capacity decides whether a payload rides in the
work-queue entry — so a setting that the runtime silently ignored would make
the compiler's plan a fiction. Every field described here is therefore either
honored, or refused by name with the reason and a pointer back to this
document. Nothing is accepted and dropped.

## 1. What is settable today

### The endpoint config

`AbolethEndpointConfig` in
[`runtime_c.h`](../include/aboleth/runtime_c.h) is the programmatic control plane.
`aboleth_runtime_create_local_endpoint()` takes one, or a null pointer for
every default.

| Field | Default | Accepted today | Effect |
|---|---|---|---|
| `struct_size` | — | `4` … `sizeof(AbolethEndpointConfig)`, a multiple of 4 | The struct's version; see below |
| `qp_type` | `0` | `0` (reliable connected) | — |
| `qp_depth` | `0` | `0`, or a send-queue work-request count the adapter can allocate | Direct transport: `ibv_qp_cap.max_send_wr` |
| `max_inline` | `0` | `0`, or requested inline bytes | Direct transport: `ibv_qp_cap.max_inline_data` |
| `qps_per_peer` | `0` | `0` or `1` | — |
| `sm_map_policy` | `0` | `0` | — |
| `port` | `0` | `0`, or the port the runtime opened | Checked against the runtime |
| `gid_index` | `-1` | `-1`, or the GID entry the runtime resolved | Checked against the runtime |
| `hca` | empty | empty, or the device the runtime opened | Checked against the runtime |

Two fields change what is built. `qp_depth` is the work-request count asked of
the adapter; the mlx5 provider sizes its ring in 64-byte blocks and rounds up,
so the queue that comes back holds at least that many requests and usually
more. What it became is
`aboleth_runtime_local_endpoint_capacity()`, in blocks — the same number the
descriptor publishes as `send_wqe_count` and the same number planning bounds a
deferral window with. The completion queue grows with it. A depth beyond the
adapter's `max_qp_wr` is refused with both numbers rather than left to
`ibv_create_qp` to answer as `ENOMEM`.

`max_inline` is a request in the same sense: the adapter may grant more, grant
less, or refuse inline data outright, in which case the endpoint falls back to
the queue pair every compiled kernel already assumes. What was granted is
`aboleth_runtime_local_endpoint_max_inline()`. The compiler plans against the
conservative floor in the capability registry, so a larger grant is an
opportunity the generated code takes, never a requirement.

Three fields are checked rather than applied. An endpoint hangs off the
`ibv_context`, protection domain and port that the runtime opened, and those
were fixed when the runtime was created — so `hca`, `port` and `gid_index`
cannot select anything here. Stating the values the runtime already holds is
accepted, because a caller that carries a topology decision through its own
code should be able to assert it. Stating anything else is refused and names
where the decision actually belongs:
`aboleth_runtime_create_for_device(gpu_index, hca_name)` for the device,
`ABOLETH_PORT` and `ABOLETH_GID_INDEX` before the runtime is created for the
other two. Partial honoring — opening a second device behind the caller's back,
or applying the port and ignoring the HCA — would leave two endpoints of one
runtime disagreeing about which fabric they are on.

Three fields exist only to be refused today: `qp_type`, `qps_per_peer` and
`sm_map_policy`. They are named in the struct because the shape of the eventual
control is already known, and because a caller that asks for one should learn
that it is deferred rather than discover later that its request did nothing.
Section 2 says where each will land.

`qp_depth` and `max_inline` apply to the direct (mlx5) transport only. A proxy
endpoint's device-visible queue is `kProxySlotCount` slots of ABI and its
inline capacity is the proxy's own; a `Transport.EFA` endpoint sizes its queues
from the libfabric endpoint. Both refuse a non-default value for either field
instead of accepting one that could not take effect. A P2P endpoint has no
queue pair at all and refuses every field.

The owned endpoint-pair entry points (`aboleth_runtime_create_endpoint_pair()`
and its proxy and EFA siblings) take no config. They exist to build both halves
of a path inside one process for the differential tests, and use the runtime's
defaults.

### Versioning by struct size

`struct_size` is the version. A caller sets it to the `sizeof` its own header
declares:

- **Smaller than this runtime's struct**: accepted. The fields the caller's
  header does not reach take this runtime's defaults, so a program compiled
  against an older header keeps working against a newer runtime, and bytes past
  the declared size are never read.
- **Equal**: accepted.
- **Larger**: refused. The caller's header is newer than the runtime it is
  calling, and the runtime cannot guess what the extra bytes mean.
- **Not a whole prefix** — zero, or a size that stops in the middle of a field
  — is refused, so a field is never half-read.

The struct is checked before the runtime handle it arrived with is looked at,
which is what lets the versioning rules be exercised on a machine with no
adapter (`test/python/test_endpoint_config.py`).

### Precedence

Every honored field has an environment variable behind it, because the
environment is how a process that never sees the struct — a launcher, a
benchmark script — configures the same machine. The order is:

**endpoint config → environment → compiled-in default.**

A config field left at its default (`0`, or `-1` for `gid_index`) falls through
to the environment; a field that is set wins over it.

| Variable | What it selects | Config field |
|---|---|---|
| `ABOLETH_HCA` | The RDMA device the runtime opens | `hca` (agreement only) |
| `ABOLETH_PORT` | The port on that device | `port` (agreement only) |
| `ABOLETH_GID_INDEX` | The GID entry the fabric is addressed through | `gid_index` (agreement only) |
| `ABOLETH_RDMA_INLINE_BYTES` | Inline capacity requested per direct endpoint | `max_inline` |
| `ABOLETH_PATH_MTU` | The negotiated path MTU ceiling | — |
| `ABOLETH_PROXY_PROVIDER` | `verbs` or `efa` behind a proxy endpoint | — |
| `ABOLETH_RDMA_SUBMIT` | `direct` or `staged`: who stores to the mlx5 UAR doorbell; unset probes the mapping and falls back ([`RDMA_ABI.md`](RDMA_ABI.md), Submission modes) | — |
| `ABOLETH_RELAXED_ORDERING`, `ABOLETH_REQUIRE_RELAXED_ORDERING` | Relaxed-ordering registration, and whether a fallback is an error | — |

`ABOLETH_GID_INDEX` is more than an escape hatch on some fabrics. On native
InfiniBand the runtime uses index 0. On RoCE the default selection returns the
first RoCE v2 entry in the GID table, which on a rail-optimized host is the
link-local `fe80::` one, and a link-local address cannot cross rails: every
cross-NIC RC connection then fails its RTR transition with a timeout. Such a
host has to set the index explicitly, and the value is whichever entry holds
the IPv4-mapped `::ffff:a.b.c.d` RoCE v2 address (`show_gids`, or
`ibv_rc_pingpong -g <index>` to confirm it carries traffic).

There is no environment variable for `qp_depth`: the send queue is sized by the
runtime and, above it, by what planning bounds a deferral window with, so a
process-wide override would change a compiled plan's premise without the
compiler seeing it.

### The Python surface

`Runtime.create_local_endpoint()` takes the honorable subset as keyword-only
arguments, all defaulted, so an existing caller is unaffected:

```python
with runtime.create_local_endpoint(
    transport="rdma",
    qp_depth=512,
    max_inline=128,
) as endpoint:
    endpoint.capacity          # blocks the adapter granted
    endpoint.max_inline_bytes  # inline bytes the adapter granted
```

`hca`, `port` and `gid_index` are accepted with the same agreement-only meaning
as in C. The three refused-only fields have no Python spelling: there is
nothing for them to select yet, and adding a keyword that can only raise would
be a surface to remove later. A caller that wants to see those refusals — or to
pass a null config — reaches the C ABI directly, as the test does.

### Fabric addressing without a runtime

A caller that opens its own queue pairs still has to answer the questions that
come before one, and those have a single answer per machine. Three exports give
the runtime's answer without creating a runtime:

- `aboleth_runtime_select_device()` — the HCA paired with a CUDA ordinal.
- `aboleth_runtime_select_gid_index()` — the GID entry for a device and port.
- `aboleth_runtime_query_port_address()` — LID, link layer, GID, GID index and
  path MTU, which is also the LID-versus-GRH decision.

Each opens and closes a short-lived `ibv_context` and retains nothing. The
direct-mlx5 probes in `experiments/cuda` are built on them
(`aboleth_probe_fabric.h`); before that they carried their own GID selection
and answered an InfiniBand fabric with a RoCE address vector.

## 2. What is deferred, and where it lands

### Queue-pair type: DC/DCT — `qp_type`

A dynamically connected queue pair replaces the per-peer RC mesh with a small
set of initiators (DCIs) that address any target (DCT), which is how a rank
scales past the point where one queue pair per peer stops fitting. NVSHMEM's
IBGDA transport exposes exactly this: `NVSHMEM_IBGDA_NUM_DCI`,
`NVSHMEM_IBGDA_NUM_SHARED_DCI` and `NVSHMEM_IBGDA_NUM_DCT`.

This is not a queue-pair creation flag. A DCI's work-queue entry carries an
address-vector segment the RC entry does not, so the device-side WQE builder,
the endpoint descriptor it reads, and the connection info exchanged with a peer
(a DCT number and access key rather than a QPN and PSN) all change with it. It
lands when the WQE builder gains a second entry shape, not before; until then
`qp_type` names the value that would select it and refuses everything but `0`.

### More than one queue pair per peer — `qps_per_peer`

Lifting several queue pairs per peer into the endpoint layer means an endpoint
handle owns a vector of queue pairs rather than one, `AbolethConnectionInfo`
carries a QPN per queue, and the compiler knows which queue a command is posted
to before it bounds a deferral window on any of them. Until then multi-CTA
code gets independent queues through separate endpoints
(`connect_all(..., endpoints_per_peer=N)`, [`WORLD.md`](WORLD.md)).

Reference designs size the mesh dynamically rather than fixing it: NVSHMEM
takes `NVSHMEM_IBGDA_NUM_RC_PER_PE`, and DeepEP derives its queue count from
the SM count and its dispatch mode. That is the shape this field should take
when it grows: a request, capped by what the adapter and the plan allow.

### SM and warp mapping onto queues — `sm_map_policy`

Once there is more than one queue pair, something has to decide which one a
block or a warp posts to. NVSHMEM makes it a policy —
`NVSHMEM_IBGDA_DCI_MAP_BY` and `NVSHMEM_IBGDA_RC_MAP_BY`, each `cta`, `sm`,
`warp` or `dct`. DeepEP computes it on the device, giving an SM its own queue
when there are more queues than SMs and sharing queues otherwise.

What is missing is the policy as an input the compiler agrees with, since the
mapping decides which queue's capacity a phase's commands consume. It follows
`qps_per_peer`; while an endpoint carries one queue pair there is nothing to
map.
