"""Endpoint profiles: the fit, the on-disk cache and the world agreement.

The fit is checked on the B200 samples the seed profile was frozen from; the
resolution runs over threads with a fake World, so no GPU is needed.
"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from aboleth.endpoint_profiles import (
    B200_NDR400_HOST,
    EndpointRegistration,
    ProfileStore,
    check_holdout,
    fit_lanes,
    fit_profile,
    fit_sender_warps,
    planned_lanes,
    resolve_endpoint_profiles,
    world_rates,
)
from aboleth.planning import (
    EndpointPartitionFacts,
    TransportPartitionPlan,
    plan_endpoint_work,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "b200_endpoint_calibration.json").read_text()
)
H100 = json.loads(
    (Path(__file__).parent / "data" / "h100_endpoint_calibration.json").read_text()
)
PLATFORM = B200_NDR400_HOST.platform


def cells(rows):
    return {(size, order, width): ranks for size, order, width, ranks in rows}


def synthetic(rate, sizes=(65536, 262144, 1 << 20, 4 << 20), widths=(1, 2, 4, 8)):
    """Rows whose rate(size, width) GB/s is given; one rank, three trials."""
    return [
        [size, order, width, [[size / (rate(size, width) * 1e3)] * 3]]
        for size in sizes
        for order in (0, 1)
        for width in widths
    ]


class FitTest(unittest.TestCase):
    def test_b200_samples_reproduce_the_seed(self):
        self.assertEqual(
            fit_lanes(world_rates(cells(FIXTURE["fit"]))),
            (FIXTURE["seed"]["bytes_per_endpoint"], FIXTURE["seed"]["max_lanes"]),
        )
        self.assertEqual(
            (B200_NDR400_HOST.bytes_per_endpoint, B200_NDR400_HOST.max_lanes),
            (262144, 4),
        )

    def test_h100_samples(self):
        # v1 (NEAR of the widest also at the crossover) gave P2P 2 MiB x 8
        # and RDMA no crossover on these samples.
        rdma, analysis = fit_profile("rdma", H100["rdma"])
        self.assertEqual(analysis["status"], "fitted")
        self.assertEqual((rdma["bytes_per_endpoint"], rdma["max_lanes"]), (16384, 4))
        p2p, analysis = fit_profile("p2p", H100["p2p"])
        self.assertEqual(analysis["status"], "fitted")
        self.assertEqual(
            (p2p["bytes_per_endpoint"], p2p["max_lanes"], p2p["sender_warps"]),
            (8192, 8, 4))

    def test_b200_holdout_passes(self):
        rows = check_holdout(world_rates(cells(FIXTURE["holdout"])), 262144, 4)
        self.assertEqual({row["lanes"] for row in rows}, {2, 4})
        self.assertTrue(all(row["passed"] for row in rows), rows)

    def test_flat_queues_fit_no_fanout(self):
        self.assertIsNone(fit_lanes(world_rates(cells(synthetic(lambda s, w: 20.0)))))

    def test_a_slow_rank_decides_each_trial(self):
        fast, slow = [[1.0, 1.0, 1.0]], [[1.0, 4.0, 4.0]]
        rates = world_rates({(1000, 0, 1): fast + slow})
        self.assertAlmostEqual(rates[1000, 0, 1], 1000 / 4.0 / 1e3)

    def test_fit_takes_the_fewest_lanes_that_saturate(self):
        # Two lanes saturate from 1 MiB; one lane carries 10 GB/s.
        def rate(size, width):
            return 10.0 * min(width, 2) if size >= 1 << 20 else 10.0
        self.assertEqual(fit_lanes(world_rates(cells(synthetic(rate)))), (1 << 19, 2))

    def test_every_order_must_hold(self):
        rows = synthetic(lambda s, w: 10.0 * w if s >= 1 << 20 else 10.0)
        for row in rows:  # the reversed sweep never gains
            if row[1] == 1:
                row[3] = [[row[0] / 10e3] * 3]
        self.assertIsNone(fit_lanes(world_rates(cells(rows))))

    def test_planned_lanes_match_plan_endpoints(self):
        partition = TransportPartitionPlan(backend="rdma", peers=(1,), grid_extent=1)
        participants = ((PLATFORM, "host", 8),) * 2
        facts = (EndpointPartitionFacts(partition, "host", PLATFORM, 8, participants),)
        for payload in (4096, 262144, 262145, 600000, 1 << 20, 64 << 20):
            (launch,) = plan_endpoint_work(
                name="calibration-check", payload_bytes=payload, max_items=1 << 20,
                partitions=facts, profiles=(B200_NDR400_HOST,), rank_count=2,
            )
            self.assertEqual(launch.lanes, planned_lanes(payload, 262144, 4), payload)

    def test_sender_warps_take_the_fewest_near_the_best(self):
        rates = world_rates(cells(synthetic(
            lambda s, w: {1: 10.0, 2: 18.0, 4: 19.5, 8: 20.0}[w])))
        self.assertEqual(fit_sender_warps(rates), 4)

    def test_failed_holdout_keeps_one_lane(self):
        samples = dict(
            fit=synthetic(lambda s, w: 10.0 * min(w, 4) if s >= 1 << 20 else 10.0),
            holdout=synthetic(lambda s, w: 10.0 / w, sizes=(2 << 20,)),
        )
        fields, analysis = fit_profile("rdma", samples)
        self.assertEqual(analysis["status"], "holdout-failed")
        self.assertEqual(fields["max_lanes"], 1)


class ThreadWorld:
    """Ranks as threads: all_gather over one barrier, a host per `hosts`."""

    def __init__(self, size, hosts):
        self.size, self.hosts = size, hosts
        self.barrier = threading.Barrier(size)
        self.lock = threading.Lock()
        self.slots = {}

    def rank(self, rank):
        state = dict(sequence=0)

        def all_gather(tag, payload):
            sequence = state["sequence"]
            state["sequence"] += 1
            with self.lock:
                self.slots.setdefault(sequence, {})[rank] = bytes(payload)
            self.barrier.wait()
            gathered = dict(self.slots[sequence])
            self.barrier.wait()
            return gathered

        return SimpleNamespace(
            rank=rank,
            world_size=self.size,
            endpoints_per_peer=8,
            bootstrap=SimpleNamespace(all_gather=all_gather),
            topology={r: SimpleNamespace(hostname=self.hosts[r]) for r in range(self.size)},
        )

    def run(self, work):
        results, errors = [None] * self.size, []

        def body(rank):
            try:
                results[rank] = work(rank, self.rank(rank))
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        if errors:
            raise errors[0]
        return results


GPU = "[\"NVIDIA H100\",[9,0],\"0x15b3\",\"0x1021\",\"28.39.1002\",\"200 Gb/sec (2X NDR)\",\"staged\"]"
REGISTRATION = EndpointRegistration(
    platform=GPU, queue_memory="gpu", participants=((GPU, "gpu", 8),) * 4,
    peer_link="NV6", participant_links=("NV6",) * 4,
)


class ResolutionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        # Two hosts, each with its own cache directory.
        self.stores = {host: ProfileStore(root / host) for host in ("a", "b")}
        self.hosts = ("a", "a", "b", "b")
        self.probes = []

    def tearDown(self):
        self.folder.cleanup()

    def measure(self, world, backends, tag):
        if world.rank == 0:
            self.probes.append(backends)
        samples = dict(fit=FIXTURE["fit"], holdout=FIXTURE["holdout"])
        result = {}
        for backend in backends:
            result[backend] = dict(samples=dict(samples), seconds=0.5, world=dict(ranks=4))
            if backend == "p2p":
                result[backend]["samples"].update(
                    warps=synthetic(lambda s, w: 10.0 * w), sender_warps=8)
        return result

    def resolve(self, **options):
        world = ThreadWorld(4, self.hosts)
        return world.run(lambda rank, w: resolve_endpoint_profiles(
            w, REGISTRATION, store=self.stores[self.hosts[rank]],
            measure=self.measure, **options))

    def test_first_contact_measures_and_later_runs_reuse_the_cache(self):
        first = self.resolve()
        self.assertEqual(len({r.identity for r in first}), 1)
        self.assertEqual([s["source"] for s in first[0].sources], ["measured", "measured"])
        rdma = next(p for p in first[0].profiles if p.backend == "rdma")
        self.assertEqual((rdma.bytes_per_endpoint, rdma.max_lanes), (262144, 4))
        p2p = next(p for p in first[0].profiles if p.backend == "p2p")
        self.assertEqual((p2p.peer_link, p2p.sender_warps), ("NV6", 8))
        # Every rank writes the same records.
        for store in self.stores.values():
            self.assertEqual(len(store.entries()), 2)
        second = self.resolve()
        self.assertEqual(len(self.probes), 1)
        self.assertEqual([s["source"] for s in second[0].sources], ["cached", "cached"])
        self.assertEqual(
            [p.identity for p in second[0].profiles],
            [p.identity for p in first[0].profiles],
        )

    def test_private_stores_per_rank_converge(self):
        # Launchers give each rank a private Triton home; a profile store
        # private to a rank must still serve the next run.
        root = Path(self.folder.name) / "private"
        stores = [ProfileStore(root / str(rank)) for rank in range(4)]

        def resolve():
            return ThreadWorld(4, self.hosts).run(lambda rank, w: resolve_endpoint_profiles(
                w, REGISTRATION, store=stores[rank], measure=self.measure))

        resolve()
        second = resolve()[0]
        self.assertEqual(len(self.probes), 1)
        self.assertEqual({s["source"] for s in second.sources}, {"cached"})

    def test_a_host_without_the_entry_recalibrates_everyone(self):
        self.resolve()
        for entry in self.stores["b"].entries():
            entry.path.unlink()
        self.resolve()
        self.assertEqual(len(self.probes), 2)

    def test_a_corrupt_entry_is_never_trusted(self):
        self.resolve()
        for store in self.stores.values():
            for entry in store.entries():
                record = json.loads(entry.path.read_text())
                record["profile"]["max_lanes"] = 16
                entry.path.write_text(json.dumps(record))
        self.assertEqual(self.stores["a"].entries(), [])
        self.assertEqual(len(self.stores["a"].scan()[1]), 2)
        self.resolve()
        self.assertEqual(len(self.probes), 2)

    def test_disabled_calibration_falls_back_to_one_lane(self):
        resolved = self.resolve(calibrate=False)[0]
        self.assertEqual(resolved.profiles, ())
        self.assertEqual({s["source"] for s in resolved.sources}, {"fallback"})
        self.assertIn("calibration disabled", resolved.identity)
        self.assertEqual(self.probes, [])

    def test_a_failed_probe_falls_back_and_says_why(self):
        def failing(world, backends, tag):
            raise RuntimeError("probe check failed on rank 2: payload mismatch")
        world = ThreadWorld(4, self.hosts)
        resolved = world.run(lambda rank, w: resolve_endpoint_profiles(
            w, REGISTRATION, store=self.stores[self.hosts[rank]], measure=failing))[0]
        self.assertEqual(resolved.profiles, ())
        self.assertIn("payload mismatch", resolved.sources[0]["reason"])

    def test_pinned_and_retired_entries(self):
        first = self.resolve()[0]
        old = next(p.identity for p in first.profiles if p.backend == "rdma")
        for store in self.stores.values():
            store.set_flags(old, retired=True)
        self.resolve()  # a retired entry is measured again
        self.assertEqual(len(self.probes), 2)
        # Reproducing the first run exactly: pin its identity.
        pinned = self.resolve(pin=(old,))[0]
        self.assertIn(f"rdma:pinned:{old}", pinned.identity)
        self.assertEqual(len(self.probes), 2)

    def test_refresh_measures_even_when_cached(self):
        self.resolve()
        self.resolve(refresh=True)
        self.assertEqual(len(self.probes), 2)

    def test_stale_protocol_recalibrates_unless_pinned(self):
        self.resolve()
        for store in self.stores.values():
            for entry in store.entries():
                record = dict(entry.record, protocol="aboleth-endpoint-probe-v0")
                body = {k: v for k, v in record.items() if k not in ("identity", "flags")}
                from aboleth.endpoint_profiles import _identity
                entry.path.unlink()
                store.write(dict(record, identity=_identity(body)))
        self.resolve()
        self.assertEqual(len(self.probes), 2)

    def test_a_seed_is_used_without_probing(self):
        seeded = EndpointRegistration(
            platform=PLATFORM, queue_memory="host",
            participants=((PLATFORM, "host", 8),) * 4,
        )
        world = ThreadWorld(4, self.hosts)
        resolved = world.run(lambda rank, w: resolve_endpoint_profiles(
            w, seeded, store=self.stores[self.hosts[rank]], measure=self.measure))[0]
        self.assertEqual(resolved.profiles, (B200_NDR400_HOST,))
        self.assertEqual(resolved.sources[0]["source"], "seed")
        self.assertEqual(self.probes, [])

    def test_mixed_platforms_get_no_lookup(self):
        mixed = EndpointRegistration(
            platform=GPU, queue_memory="gpu",
            participants=((GPU, "gpu", 8), (PLATFORM, "host", 8)),
        )
        self.assertEqual(mixed.profile_lookups(), ())


if __name__ == "__main__":
    unittest.main()
