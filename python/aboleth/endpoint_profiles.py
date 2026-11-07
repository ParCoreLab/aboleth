"""Endpoint cost facts the runtime reports; no application-specific policy.

A profile states how many bytes one queue serves well and how many lanes are
worth using on one platform (GPU, NIC, firmware, link, submission mode, queue
memory). plan-endpoints decides lanes from it. The runtime measures it once
per platform (`endpoint_probe`), fits it here, agrees on it across ranks and
caches it on disk; `python -m aboleth.endpoint_profiles` lists, shows, pins
and retires the cached entries.

The B200 entries are seeds, shipped like pinned entries: B200/NDR400 direct
mlx5 with host-resident queues, calibrated and then confirmed on independent
sizes. The four-lane candidate was frozen before the independent sizes were
run; the fit below reproduces it from the calibration samples
(test/python/data/b200_endpoint_calibration.json).
Unknown GPU, NIC firmware, link mode or queue memory does not match this key.
"""

import hashlib
import json
import os
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .planning import EndpointProfile


def endpoint_platform_key(
    *,
    gpu_name: str,
    compute_capability: tuple[int, int],
    nic_vendor: str,
    nic_device: str,
    nic_firmware: str,
    link_rate: str,
    submission: str,
) -> str:
    """Canonical exact registration facts, independent of UUID and hostname."""
    import json

    return json.dumps(
        [
            gpu_name,
            list(compute_capability),
            nic_vendor,
            nic_device,
            nic_firmware,
            link_rate,
            submission,
        ],
        separators=(",", ":"),
    )


B200_NDR400_HOST = EndpointProfile(
    backend="rdma",
    platform=endpoint_platform_key(
        gpu_name="NVIDIA B200",
        compute_capability=(10, 0),
        nic_vendor="0x15b3",
        nic_device="0x1021",
        nic_firmware="28.47.2682",
        link_rate="400 Gb/sec (4X NDR)",
        submission="direct",
    ),
    queue_memory="host",
    bytes_per_endpoint=262144,
    max_lanes=4,
    identity="sha256:11294cae94126dd1d3eb4d4eff4f5e960ce5304daf772eae1014a7554f2265ee",
)
# Prospective fixed policy and independent-size confirmation, on both NV18
# B200 nodes: phase3-prefix-copy-diagnosis/phase3-p2p-cost-holdout-v1/analysis.json.
# The same lanes with the accepted copy's fixed 256-thread block geometry are
# independently confirmed in phase3-p2p-geometry-holdout-v1/analysis.json.
# This restores parallel local copies; it is not a per-model resource search.
B200_NV18 = EndpointProfile(
    backend="p2p",
    platform=B200_NDR400_HOST.platform,
    queue_memory="none",
    bytes_per_endpoint=262144,
    max_lanes=16,
    peer_link="NV18",
    sender_warps=8,
    identity="sha256:9ff9b82e2f66c3689035f6325e0ce0d87e053856e8ab1faa7a151ca221a5f1c5",
)
# MI355X VF, direct KFD XGMI EP2. Frozen policy8ec74500: application-independent
# record calibration H1024,N2/16/64/256, then independent H1536,N1/6/24/96.
# No AMD/native MoE workload outcome was used to choose this geometry.
# The platform key includes actual GPU/firmware, HIP and amdgpu versions.
AMD_MI355X_VF_XGMI_EP2 = EndpointProfile(
    backend="p2p",
    platform='["amd-kfd-p2p-v1","AMD Instinct MI355X VF","gfx950:sramecc+:xnack-","7.14.60850","6.19.14.31400000",{"device_id":30131,"fw_version":44,"gfx_target_version":90500,"num_xcc":8,"vendor_id":4098,"wave_front_size":64}]',
    queue_memory="none",
    bytes_per_endpoint=16384,
    max_lanes=16,
    peer_link='["amd-kfd-direct-link-v1",{"flags":1,"max_bandwidth":76000,"min_bandwidth":76000,"type":11,"weight":15}]',
    sender_warps=8,
    identity='sha256:8ec7450091945f66d8aa3f05256fdf3c982e701fbc94d13dd7e36c87021eff21',
)
QUALIFIED_ENDPOINT_PROFILES = (B200_NDR400_HOST, B200_NV18, AMD_MI355X_VF_XGMI_EP2)


def registered_peer_link(
    device_uuid, peer_uuids, *, uuid_report=None, topology_report=None
):
    """Read the actual device UUIDs' common local link, once at registration.

    CUDA ordinals may be masked or reordered; topology GPU indices are physical.
    Missing devices, malformed reports and mixed links return an unqualified fact.
    The C++ planner still checks the canonical world image before using a profile.
    """
    import re
    import subprocess

    def uuid(value):
        match = re.search(
            r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", str(value), re.I
        )
        return match.group().lower() if match else ""

    local, peers = uuid(device_uuid), tuple(
        uuid(value) for value in peer_uuids
    )
    if not local or not peers or not all(peers) or local in peers:
        return ""
    try:
        if uuid_report is None:
            uuid_report = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=10,
            )
        if topology_report is None:
            topology_report = subprocess.check_output(
                ["nvidia-smi", "topo", "-m"], text=True, timeout=10
            )
        indices = {}
        for line in uuid_report.splitlines():
            index, identity = line.split(",", 1)
            normalized = uuid(identity)
            if normalized in indices or not normalized:
                return ""
            indices[normalized] = "GPU" + str(int(index.strip()))
        # Some driver versions underline the header even through a pipe.
        topology_report = re.sub(
            r"\x1b\[[0-?]*[ -/]*[@-~]", "", topology_report
        )
        lines = [line.split() for line in topology_report.splitlines()]
        header = next(
            row
            for row in lines
            if row
            and row[0].startswith("GPU")
            and "X" not in row
            and all(cell.startswith("GPU") for cell in row[:2])
        )
        rows = {
            row[0]: row[1:]
            for row in lines
            if row and row[0].startswith("GPU") and "X" in row
        }
        own = indices[local]
        links = {rows[own][header.index(indices[peer])] for peer in peers}
        return links.pop() if len(links) == 1 else ""
    except (
        OSError,
        subprocess.SubprocessError,
        KeyError,
        IndexError,
        ValueError,
        StopIteration,
    ):
        return ""


# ---------------------------------------------------------------------------
# Fit. Rates are useful GB/s per (bytes, order, width) cell, where width is a
# lane count (or a warp count for the P2P sender) and order indexes the
# forward and reversed sweep of widths within one size.

# A change to the probe or the fit is a new protocol; entries measured under
# another protocol are stale unless pinned.
PROBE_PROTOCOL = "aboleth-endpoint-probe-v2"
# The thresholds the B200 calibration froze (four-lane proposal and
# holdout): a wider set of queues must buy GAIN over one queue and the chosen
# width must reach NEAR of the widest measured, in every order; on held-out
# sizes the planned lanes must keep HOLDOUT of one queue. v2
# no longer asks NEAR of the widest at the crossover, only at the top: on
# NVLink, where 16 lanes stay a few percent ahead of 8 at every size, v1 put
# the crossover at the largest size and planned one lane for 1 MiB chains
# that 8 lanes copy 5x faster. Both agree on the B200 samples.
GAIN, NEAR, HOLDOUT = 1.15, 0.95, 0.90
# Widths the probe measures, capped by the endpoints the world registered.
PROBE_LANES = {"rdma": (1, 2, 4, 8), "p2p": (1, 2, 4, 8, 16)}
PROBE_WARPS = (1, 2, 4, 8)


def world_rates(cells):
    """Useful GB/s per cell from trial-aligned per-rank times.

    `cells` maps (bytes, order, width) to one list of microseconds per
    iteration per rank. A trial lasts as long as its slowest rank, and a
    cell's rate is its median trial.
    """
    rates = {}
    for (size, order, width), ranks in cells.items():
        trials = [max(times) for times in zip(*ranks)]
        if not trials or min(trials) <= 0:
            raise ValueError("calibration cell has no positive trials")
        rates[int(size), int(order), int(width)] = (
            size / statistics.median(trials) / 1e3
        )
    return rates


def _axes(rates):
    sizes, orders, widths = (sorted({key[i] for key in rates}) for i in range(3))
    if len(rates) != len(sizes) * len(orders) * len(widths):
        raise ValueError("calibration cells do not form a complete grid")
    return sizes, orders, widths


def fit_lanes(rates):
    """(bytes per endpoint, lane limit), or None when no size pays for a lane.

    The efficient width is the fewest lanes within NEAR of the widest at the
    largest size where the widest still buys GAIN over one lane. The
    crossover is the smallest size where that width buys GAIN over one lane;
    each lane then serves crossover / width bytes. Every test holds in every
    order, and a tie takes the smaller width. The crossover is never below
    the smallest probed size.
    """
    sizes, orders, lanes = _axes(rates)
    widest = lanes[-1]
    if lanes[0] != 1 or widest == 1:
        return None
    paying = [
        size
        for size in sizes
        if all(rates[size, o, widest] >= GAIN * rates[size, o, 1] for o in orders)
    ]
    if not paying:
        return None
    top = paying[-1]
    width = next(
        w
        for w in lanes
        if all(rates[top, o, w] >= NEAR * rates[top, o, widest] for o in orders)
    )
    crossover = next(
        (
            size
            for size in sizes
            if all(rates[size, o, width] >= GAIN * rates[size, o, 1] for o in orders)
        ),
        None,
    )
    return None if crossover is None else (crossover // width, width)


def planned_lanes(payload_bytes, bytes_per_endpoint, max_lanes):
    """The lanes plan-endpoints gives a chain of `payload_bytes` alone."""
    needed = -(-payload_bytes // bytes_per_endpoint)
    cap = 1 << (max_lanes.bit_length() - 1)
    lanes = 1
    while lanes < needed and lanes < cap:
        lanes *= 2
    return lanes


def check_holdout(rates, bytes_per_endpoint, max_lanes):
    """Rows for sizes the fit never saw: planned lanes against one lane."""
    rows = []
    for size, order in sorted({key[:2] for key in rates}):
        lanes = planned_lanes(size, bytes_per_endpoint, max_lanes)
        if (size, order, lanes) not in rates or (size, order, 1) not in rates:
            raise ValueError(f"holdout lacks {lanes} or one lane at {size} bytes")
        ratio = rates[size, order, lanes] / rates[size, order, 1]
        rows.append(
            dict(bytes=size, order=order, lanes=lanes, ratio_to_one=ratio,
                 passed=ratio >= HOLDOUT)
        )
    return rows


def fit_sender_warps(rates):
    """Fewest warps whose one-lane copy reaches NEAR of the best everywhere.

    When no warp count does, the one with the best worst-case share.
    """
    sizes, orders, warps = _axes(rates)

    def share(w):
        return min(
            rates[s, o, w] / max(rates[s, o, v] for v in warps)
            for s in sizes
            for o in orders
        )

    return next((w for w in warps if share(w) >= NEAR), max(warps, key=share))


def _cells(rows):
    return {(size, order, width): ranks for size, order, width, ranks in rows}


def fit_profile(backend, samples):
    """Profile fields (without platform facts) and the analysis of a probe.

    `samples` holds trial-aligned world rows [bytes, order, width, per-rank
    microseconds]: "fit" and "holdout" over lanes, and for P2P "warps" over
    one-lane sender warps with the lanes measured at "sender_warps". A fit
    without a crossover, or one whose holdout fails, keeps one lane.
    """
    rates = world_rates(_cells(samples["fit"]))
    fitted = fit_lanes(rates)
    analysis = dict(rule=dict(gain=GAIN, near=NEAR, holdout=HOLDOUT))
    fields = dict(backend=backend, sender_warps=1)
    if backend == "p2p":
        warps = world_rates(_cells(samples["warps"]))
        fields["sender_warps"] = fit_sender_warps(warps)
        if fields["sender_warps"] != samples["sender_warps"]:
            raise ValueError("P2P lanes were measured at other sender warps")
        analysis["warps_rates"] = sorted([*k, v] for k, v in warps.items())
    analysis["rates"] = sorted([*k, v] for k, v in rates.items())
    largest = max(key[0] for key in rates)
    if fitted is None:
        analysis["status"] = "no-crossover"
        fields.update(bytes_per_endpoint=largest, max_lanes=1)
        return fields, analysis
    bytes_per_endpoint, width = fitted
    holdout = check_holdout(
        world_rates(_cells(samples["holdout"])), bytes_per_endpoint, width
    )
    analysis["holdout"] = holdout
    analysis["fit"] = dict(bytes_per_endpoint=bytes_per_endpoint, max_lanes=width)
    if not all(row["passed"] for row in holdout):
        analysis["status"] = "holdout-failed"
        fields.update(bytes_per_endpoint=largest, max_lanes=1)
        return fields, analysis
    analysis["status"] = "fitted"
    fields.update(bytes_per_endpoint=bytes_per_endpoint, max_lanes=width)
    return fields, analysis


# ---------------------------------------------------------------------------
# Cache. One immutable file per measurement; only its flags change. An entry
# is looked up by (backend, platform, queue memory, peer link).

SEED_PROFILES = QUALIFIED_ENDPOINT_PROFILES
_PROFILE_FIELDS = (
    "backend", "platform", "queue_memory", "bytes_per_endpoint", "max_lanes",
    "peer_link", "sender_warps",
)


def _identity(body):
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _lookup(fields):
    return (fields["backend"], fields["platform"], fields["queue_memory"],
            fields.get("peer_link", ""))


def default_profile_root():
    """The user's cache directory, shared by every rank and run.

    Not beside Triton's cache: launchers give each rank a private
    TRITON_CACHE_DIR and TRITON_HOME (scripts/multinode-launch.sh), and a
    profile must be one fact for the whole world.
    """
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "aboleth" / "endpoint-profiles"


@dataclass(frozen=True)
class ProfileEntry:
    record: dict
    path: Path | None = None
    source: str = "cached"

    @property
    def identity(self):
        return self.record["identity"]

    @property
    def fields(self):
        return self.record["profile"]

    @property
    def lookup(self):
        return _lookup(self.fields)

    @property
    def flags(self):
        return self.record.get("flags", {})

    @property
    def pinned(self):
        return bool(self.flags.get("pinned"))

    @property
    def retired(self):
        return bool(self.flags.get("retired"))

    @property
    def stale(self):
        return self.record.get("protocol") != PROBE_PROTOCOL and not self.pinned

    @property
    def probe_width(self):
        """Widest lane set the probe had; a seed is not limited by a probe."""
        rows = self.record.get("samples", {}).get("fit")
        return max(row[2] for row in rows) if rows else None

    @property
    def profile(self):
        return EndpointProfile(identity=self.identity, **self.fields)


def _seed_entry(profile):
    fields = {name: getattr(profile, name) for name in _PROFILE_FIELDS}
    return ProfileEntry(
        dict(identity=profile.identity, profile=fields, flags=dict(pinned=True),
             protocol="seed"),
        source="seed",
    )


SEED_ENTRIES = tuple(_seed_entry(profile) for profile in SEED_PROFILES)


def _check_record(record):
    body = {k: v for k, v in record.items() if k not in ("identity", "flags")}
    if record.get("identity") != _identity(body):
        raise ValueError("identity does not match the record")
    fields = record["profile"]
    if set(fields) != set(_PROFILE_FIELDS):
        raise ValueError("profile fields are incomplete")
    if not all(isinstance(fields[k], str) for k in _PROFILE_FIELDS[:3] + ("peer_link",)):
        raise ValueError("profile facts must be strings")
    if not all(type(fields[k]) is int and fields[k] > 0
               for k in ("bytes_per_endpoint", "max_lanes", "sender_warps")):
        raise ValueError("profile sizes must be positive integers")


class ProfileStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_profile_root()

    def scan(self):
        """(valid entries, [(path, reason)] for files that cannot be trusted)."""
        valid, rejected = [], []
        for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else ():
            try:
                record = json.loads(path.read_text())
                _check_record(record)
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append((path, f"{type(error).__name__}: {error}"))
                continue
            valid.append(ProfileEntry(record, path))
        return valid, rejected

    def entries(self):
        return self.scan()[0]

    def find(self, identity):
        """The stored or seed entry whose identity starts with `identity`."""
        matches = [
            entry
            for entry in (*self.entries(), *SEED_ENTRIES)
            if entry.identity.startswith(identity)
            or entry.identity.removeprefix("sha256:").startswith(identity)
        ]
        if len({entry.identity for entry in matches}) > 1:
            raise LookupError(f"profile identity {identity!r} is ambiguous")
        return matches[0] if matches else None

    def current(self, lookup):
        """A pinned entry, else the newest fresh one, else a seed, else None."""
        candidates = [e for e in self.entries() if e.lookup == lookup and not e.retired]
        newest = sorted(candidates, key=lambda e: e.record.get("measured_utc", ""))
        pinned = [e for e in newest if e.pinned]
        fresh = [e for e in newest if not e.stale]
        if pinned or fresh:
            return (pinned or fresh)[-1]
        return next((e for e in SEED_ENTRIES if e.lookup == lookup), None)

    def write(self, record):
        _check_record(record)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (record["identity"].removeprefix("sha256:")[:32] + ".json")
        # Ranks on several hosts may share the directory (and a pid).
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return ProfileEntry(record, path)

    def set_flags(self, identity, **flags):
        entry = self.find(identity)
        if entry is None or entry.path is None:
            raise LookupError(f"no stored profile {identity!r}")
        record = dict(entry.record, flags={**entry.flags, **flags})
        return self.write(record)


def measured_record(fields, analysis, samples, *, platform, queue_memory,
                    peer_link, measured_utc, code, world, seconds):
    profile = dict(fields, platform=platform, queue_memory=queue_memory,
                   peer_link=peer_link)
    body = dict(format=1, protocol=PROBE_PROTOCOL, profile=profile,
                measured_utc=measured_utc, code=code, world=world,
                seconds=seconds, analysis=analysis, samples=samples)
    return dict(body, identity=_identity(body), flags={})


# ---------------------------------------------------------------------------
# Resolution: one profile per lookup for the whole world.


@dataclass(frozen=True)
class EndpointRegistration:
    """Registered endpoint facts of every rank, in rank order."""

    platform: str
    queue_memory: str
    participants: tuple[tuple[str, str, int], ...]
    peer_link: str = ""
    participant_links: tuple[str, ...] = ()

    def plan_facts(self):
        """Keyword facts for compile_moe_plan / EndpointPartitionFacts."""
        return asdict(self)

    def profile_lookups(self):
        """Profiles the planner can use for this world; identical on every rank.

        A profile applies only when every rank reports the same platform (and
        the same peer link for P2P), as plan-endpoints requires.
        """
        lookups = []
        platforms = {(p, m) for p, m, _ in self.participants}
        if len(platforms) == 1:
            platform, memory = next(iter(platforms))
            if platform and memory != "none":
                lookups.append(("rdma", platform, memory, ""))
            links = set(self.participant_links)
            if platform and len(links) == 1 and "" not in links:
                lookups.append(("p2p", platform, "none", next(iter(links))))
        return tuple(lookups)


@dataclass(frozen=True)
class ResolvedProfiles:
    """The profiles a world plans with, and where each came from."""

    profiles: tuple[EndpointProfile, ...] = ()
    # One per lookup: backend, source (pinned, seed, cached, measured or
    # fallback), identity, reason, and the probe's seconds when it measured.
    sources: tuple[dict, ...] = field(default=())
    seconds: float = 0.0

    @property
    def identity(self):
        """What compile keys and run records name: sources and identities."""
        return ",".join(
            f"{s['backend']}:{s['source']}:{s.get('identity') or s.get('reason', '')}"
            for s in self.sources
        ) or "none"

    def record(self):
        return dict(identity=self.identity, sources=list(self.sources),
                    seconds=self.seconds,
                    profiles=[asdict(profile) for profile in self.profiles])


def _code_version():
    import subprocess

    commit = os.environ.get("ABOLETH_COMMIT", "")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
    return dict(commit=commit, protocol=PROBE_PROTOCOL)


def resolve_endpoint_profiles(world, registration, *, pin=(), calibrate=True,
                              refresh=False, store=None, measure=None,
                              tag="aboleth.endpoint-profiles.v1"):
    """Collective: the same profiles on every rank, measured if none agree.

    Each rank looks up a pinned, cached or seed entry per lookup. When every
    rank names the same entry, that is the profile. Otherwise (or with
    `refresh`) the ranks probe together, fit the world's samples
    identically, and cache the result. A
    probe that is disabled, impossible or fails leaves the lookup without a
    profile (one lane), and the record says why.

    Every rank writes the agreed record (identical bytes, atomic replace), so
    stores private to a rank or a host converge as well as a shared one.

    `pin` names stored identities that must be used exactly, on every rank.
    `measure(world, backends, tag)` is the probe; the default is
    endpoint_probe.measure.
    """
    import time

    from .compiler import enabled_optimizations

    began = time.monotonic()
    store = ProfileStore() if store is None else store
    lookups = registration.profile_lookups()
    pins = {}
    for identity in pin:
        entry = store.find(identity)
        if entry is None:
            raise LookupError(f"pinned profile {identity!r} is not stored here")
        pins[entry.lookup] = ProfileEntry(entry.record, entry.path, "pinned")
    local = []
    for lookup in lookups:
        entry = pins.get(lookup) or store.current(lookup)
        if (entry is not None and entry.source == "cached"
                and entry.probe_width is not None
                and entry.probe_width < min(world.endpoints_per_peer,
                                            _probe_limit(lookup[0]))):
            entry = None  # measured with fewer lanes than this world has
        if entry is not None and entry.source == "cached" and entry.pinned:
            entry = ProfileEntry(entry.record, entry.path, "pinned")
        if refresh and lookup not in pins:
            entry = None
        local.append(entry)
    image = [registration.platform, [e.identity if e else "" for e in local],
             _code_version(), datetime.now(timezone.utc).isoformat()]
    gathered = world.bootstrap.all_gather(tag + ".lookup", json.dumps(image).encode())
    images = [json.loads(gathered[rank]) for rank in range(world.world_size)]
    sources, profiles, missing = [], [], []
    for index, lookup in enumerate(lookups):
        names = {img[1][index] for img in images}
        if len(names) == 1 and "" not in names:
            entry = local[index]
            sources.append(dict(backend=lookup[0], source=entry.source,
                                identity=entry.identity))
            profiles.append(entry.profile)
        elif lookup in pins:
            raise RuntimeError("a pinned profile is not used on every rank")
        else:
            missing.append(lookup)
    reason = None
    if missing and not calibrate:
        reason = "calibration disabled"
    elif missing and "endpoint-fanout" not in enabled_optimizations():
        reason = "endpoint-fanout disabled"
    measured = {}
    if missing and reason is None:
        if measure is None:
            from .endpoint_probe import measure
        try:
            measured = measure(world, tuple(l[0] for l in missing), tag + ".probe")
        except Exception as error:  # noqa: BLE001
            # The probe agrees on failure collectively before raising.
            reason = f"probe failed: {type(error).__name__}: {error}"
    for lookup in missing:
        result = measured.get(lookup[0])
        why = reason or (result or {}).get("reason") or "probe measured nothing"
        if result is None or "samples" not in result:
            sources.append(dict(backend=lookup[0], source="fallback", reason=why))
            continue
        try:
            fields, analysis = fit_profile(lookup[0], result["samples"])
        except (ValueError, KeyError) as error:
            sources.append(dict(backend=lookup[0], source="fallback",
                                reason=f"fit failed: {error}"))
            continue
        record = measured_record(
            fields, analysis, result["samples"], platform=lookup[1],
            queue_memory=lookup[2], peer_link=lookup[3],
            measured_utc=images[0][3], code=images[0][2],
            world=result["world"], seconds=result["seconds"],
        )
        entry = store.write(record)
        sources.append(dict(backend=lookup[0], source="measured",
                            identity=entry.identity, status=analysis["status"],
                            seconds=result["seconds"]))
        profiles.append(entry.profile)
    resolved = ResolvedProfiles(tuple(profiles), tuple(sources),
                                time.monotonic() - began)
    agreed = world.bootstrap.all_gather(tag + ".agreed", resolved.identity.encode())
    if len(set(agreed.values())) != 1:
        raise RuntimeError("ranks resolved different endpoint profiles")
    return resolved


def _probe_limit(backend):
    return PROBE_LANES[backend][-1]


# ---------------------------------------------------------------------------
# Command line.


def _describe(entry):
    fields = entry.fields
    state = ("pinned " if entry.pinned else "") + ("retired " if entry.retired else "") + (
        "stale " if entry.stale else "")
    return (f"{entry.identity[:19]}  {entry.source:6} {fields['backend']:4} "
            f"{fields['bytes_per_endpoint']:>9} B x{fields['max_lanes']:<2} "
            f"warps {fields['sender_warps']}  {entry.record.get('measured_utc', '-')[:19]}  "
            f"{state}{fields['platform']} {fields['queue_memory']} {fields['peer_link']}")


def _calibrate(args, store):
    import torch

    from .bootstrap import TcpStoreBootstrap
    from .world import World

    rank = int(os.environ["RANK"])
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    bootstrap = TcpStoreBootstrap.from_env(timeout=300)
    world = World(rank, bootstrap.world_size, device=device, bootstrap=bootstrap)
    try:
        world.connect_all(args.transport, endpoints_per_peer=args.lanes)
        resolved = world.endpoint_profiles(refresh=True, store=store)
        if rank == 0:
            print(json.dumps(resolved.record(), indent=1))
            for source in resolved.sources:
                entry = store.find(source.get("identity") or "-") if source.get("identity") else None
                seed = next((s for s in SEED_ENTRIES if entry and s.lookup == entry.lookup), None)
                if seed is not None:
                    print(f"seed {seed.identity[:19]}: {seed.fields['bytes_per_endpoint']} B "
                          f"x{seed.fields['max_lanes']}, measured "
                          f"{entry.fields['bytes_per_endpoint']} B x{entry.fields['max_lanes']}")
    finally:
        world.close()
        bootstrap.close()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m aboleth.endpoint_profiles",
        description="Measured endpoint profiles cached per platform.")
    parser.add_argument("--root", type=Path, help=f"default: {default_profile_root()}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="stored, seed and rejected entries")
    for name, text in [("show", "one entry as JSON"), ("pin", "use this entry for its platform, even when stale"),
                       ("unpin", "drop the pin"),
                       ("recalibrate", "retire this entry; the next contact measures again")]:
        commands.add_parser(name, help=text).add_argument("identity")
    calibrate = commands.add_parser(
        "calibrate", help="measure now, under a launcher (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)")
    calibrate.add_argument("--transport", default="auto")
    calibrate.add_argument("--lanes", type=int, default=16, help="endpoints per peer to connect")
    args = parser.parse_args(argv)
    store = ProfileStore(args.root)
    if args.command == "list":
        valid, rejected = store.scan()
        for entry in (*valid, *SEED_ENTRIES):
            print(_describe(entry))
        for path, why in rejected:
            print(f"rejected {path}: {why}")
        return 0
    if args.command == "calibrate":
        _calibrate(args, store)
        return 0
    try:
        entry = store.find(args.identity)
    except LookupError as error:
        parser.error(str(error))
    if entry is None:
        parser.error(f"no profile {args.identity!r}")
    if args.command == "show":
        print(json.dumps(entry.record, indent=1, sort_keys=True))
    elif entry.source == "seed":
        parser.error("seed profiles ship with the code and are always pinned")
    elif args.command == "pin":
        print(_describe(store.set_flags(entry.identity, pinned=True)))
    elif args.command == "unpin":
        print(_describe(store.set_flags(entry.identity, pinned=False)))
    else:
        print(_describe(store.set_flags(entry.identity, retired=True, pinned=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
