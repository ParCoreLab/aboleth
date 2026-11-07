import contextlib
import multiprocessing
import os
import socket
import time
import unittest
from unittest import mock

from aboleth.bootstrap import (
    TcpStoreBootstrap,
    decode_bundle,
    encode_bundle,
)


HOST = "127.0.0.1"
WORLD_SIZE = 4
TIMEOUT = 30.0
# The missing-peer case waits out its own budget, so it stays short: the
# process rendezvous keeps interpreter startup out of every store timeout.
MISSING_TIMEOUT = 3.0
IDLE_RANK = WORLD_SIZE - 1
RENDEZVOUS_TIMEOUT = 60
RESULT_TIMEOUT = 120


def free_port():
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def rendezvous(barrier):
    try:
        barrier.wait(timeout=RENDEZVOUS_TIMEOUT)
    except Exception:
        pass


@contextlib.contextmanager
def rank_bootstrap(
    rank,
    barrier,
    port,
    *,
    timeout=TIMEOUT,
    from_env=False,
):
    # Rank 0 hosts the store, so the ranks attach together and none of them
    # drops its store until they are all done: an early close resets the
    # peers' connections, and a late attach eats the store timeout.
    rendezvous(barrier)
    if from_env:
        settings = {
            "MASTER_ADDR": HOST,
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
        }
        with mock.patch.dict(os.environ, settings):
            bootstrap = TcpStoreBootstrap.from_env(
                timeout=timeout
            )
    else:
        bootstrap = TcpStoreBootstrap(
            rank,
            WORLD_SIZE,
            host=HOST,
            port=port,
            timeout=timeout,
        )
    try:
        yield bootstrap
    finally:
        rendezvous(barrier)
        bootstrap.close()


def run_worker(worker, rank, barrier, queue, arguments):
    try:
        result = (rank, True, worker(rank, barrier, *arguments))
    except Exception as error:
        result = (
            rank,
            False,
            f"{type(error).__name__}: {error}",
        )
    queue.put(result)


def run_ranks(worker, *arguments):
    # One process per rank, each with its own store client: nothing is shared
    # but the TCP endpoint and the rendezvous.
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    barrier = context.Barrier(WORLD_SIZE)
    processes = [
        context.Process(
            target=run_worker,
            args=(worker, rank, barrier, queue, arguments),
        )
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    results = {}
    try:
        for _ in range(WORLD_SIZE):
            rank, succeeded, value = queue.get(
                timeout=RESULT_TIMEOUT
            )
            results[rank] = (succeeded, value)
    finally:
        for process in processes:
            process.join(RESULT_TIMEOUT)
            if process.is_alive():
                process.terminate()
                process.join()
    return results


def decoded(gathered):
    return {
        peer: payload.decode()
        for peer, payload in gathered.items()
    }


def gather_worker(rank, barrier, port):
    with rank_bootstrap(rank, barrier, port) as bootstrap:
        return decoded(
            bootstrap.all_gather(
                "aboleth.test.topology",
                f"rank-{rank}".encode(),
            )
        )


def repeated_worker(rank, barrier, port):
    with rank_bootstrap(rank, barrier, port) as bootstrap:
        return [
            decoded(
                bootstrap.all_gather(
                    "aboleth.test.repeated",
                    f"round-{index}-rank-{rank}".encode(),
                )
            )
            for index in range(3)
        ]


def two_tag_worker(rank, barrier, port):
    with rank_bootstrap(rank, barrier, port) as bootstrap:
        first = bootstrap.all_gather(
            "aboleth.test.first",
            f"first-{rank}".encode(),
        )
        second = bootstrap.all_gather(
            "aboleth.test.second",
            f"second-{rank}".encode(),
        )
        return decoded(first), decoded(second)


def environment_worker(rank, barrier, port):
    with rank_bootstrap(
        rank,
        barrier,
        port,
        from_env=True,
    ) as bootstrap:
        gathered = bootstrap.all_gather(
            "aboleth.test.from-env",
            f"env-{rank}".encode(),
        )
        return (
            bootstrap.rank,
            bootstrap.world_size,
            decoded(gathered),
        )


def barrier_worker(rank, barrier, port):
    with rank_bootstrap(rank, barrier, port) as bootstrap:
        bootstrap.barrier()
        bootstrap.barrier("aboleth.test.barrier")
        return "reached"


def missing_peer_worker(rank, barrier, port):
    # Every rank attaches, so rank 0 stops waiting for workers, but one of
    # them never reaches the collective.
    with rank_bootstrap(
        rank,
        barrier,
        port,
        timeout=MISSING_TIMEOUT,
    ) as bootstrap:
        if rank == IDLE_RANK:
            time.sleep(MISSING_TIMEOUT + 1.0)
            return "idle"
        try:
            bootstrap.all_gather(
                "aboleth.test.missing",
                f"rank-{rank}".encode(),
            )
        except TimeoutError as error:
            return str(error)
        return "gathered"


class TcpStoreBootstrapTest(unittest.TestCase):
    def gathered_results(self, results):
        self.assertEqual(set(results), set(range(WORLD_SIZE)))
        for rank, (succeeded, value) in sorted(
            results.items()
        ):
            self.assertTrue(
                succeeded,
                f"rank {rank} failed: {value}",
            )
        return {
            rank: value
            for rank, (_, value) in results.items()
        }

    def test_all_gather_returns_every_rank(self):
        results = self.gathered_results(
            run_ranks(gather_worker, free_port())
        )
        expected = {
            peer: f"rank-{peer}"
            for peer in range(WORLD_SIZE)
        }
        for rank in range(WORLD_SIZE):
            self.assertEqual(results[rank], expected)

    def test_repeated_tag_gathers_fresh_payloads(self):
        results = self.gathered_results(
            run_ranks(repeated_worker, free_port())
        )
        for rank in range(WORLD_SIZE):
            rounds = results[rank]
            self.assertEqual(len(rounds), 3)
            for index, gathered in enumerate(rounds):
                self.assertEqual(
                    gathered,
                    {
                        peer: f"round-{index}-rank-{peer}"
                        for peer in range(WORLD_SIZE)
                    },
                )

    def test_two_tags_in_sequence(self):
        results = self.gathered_results(
            run_ranks(two_tag_worker, free_port())
        )
        for rank in range(WORLD_SIZE):
            first, second = results[rank]
            self.assertEqual(
                first,
                {
                    peer: f"first-{peer}"
                    for peer in range(WORLD_SIZE)
                },
            )
            self.assertEqual(
                second,
                {
                    peer: f"second-{peer}"
                    for peer in range(WORLD_SIZE)
                },
            )

    def test_from_env_reads_the_launcher_variables(self):
        results = self.gathered_results(
            run_ranks(environment_worker, free_port())
        )
        for rank in range(WORLD_SIZE):
            self.assertEqual(
                results[rank],
                (
                    rank,
                    WORLD_SIZE,
                    {
                        peer: f"env-{peer}"
                        for peer in range(WORLD_SIZE)
                    },
                ),
            )

    def test_barrier_completes(self):
        results = self.gathered_results(
            run_ranks(barrier_worker, free_port())
        )
        for rank in range(WORLD_SIZE):
            self.assertEqual(results[rank], "reached")

    def test_missing_peer_times_out(self):
        results = self.gathered_results(
            run_ranks(missing_peer_worker, free_port())
        )
        self.assertEqual(results[IDLE_RANK], "idle")
        for rank in range(WORLD_SIZE):
            if rank == IDLE_RANK:
                continue
            message = results[rank]
            self.assertIn(
                f"rank {rank} did not receive",
                message,
            )
            self.assertIn(f"rank {IDLE_RANK}", message)
            self.assertIn("aboleth.test.missing", message)


class TcpStoreBootstrapEnvironmentTest(unittest.TestCase):
    def test_from_env_names_the_missing_variable(self):
        settings = {
            "MASTER_ADDR": HOST,
            "RANK": "0",
            "WORLD_SIZE": str(WORLD_SIZE),
        }
        with mock.patch.dict(
            os.environ,
            settings,
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "MASTER_PORT is unset",
            ):
                TcpStoreBootstrap.from_env()

    def test_close_is_idempotent(self):
        bootstrap = TcpStoreBootstrap(
            0,
            1,
            host=HOST,
            port=free_port(),
            timeout=MISSING_TIMEOUT,
        )
        self.assertEqual(
            bootstrap.all_gather("aboleth.test.solo", b"only"),
            {0: b"only"},
        )
        bootstrap.close()
        bootstrap.close()
        with self.assertRaisesRegex(
            RuntimeError,
            "bootstrap is closed",
        ):
            bootstrap.all_gather("aboleth.test.solo", b"only")

    def test_rank_must_be_inside_the_world(self):
        with self.assertRaisesRegex(
            ValueError,
            "outside the world",
        ):
            TcpStoreBootstrap(
                WORLD_SIZE,
                WORLD_SIZE,
                host=HOST,
                port=free_port(),
            )


class BundleFramingTest(unittest.TestCase):
    # The framing every World exchange rides on: what a record holds is the
    # exchange's business, where one ends is this.
    RECORDS = (b"", b"one", bytes(range(256)))

    def test_records_survive_the_round_trip(self):
        encoded = encode_bundle(3, 7, self.RECORDS)
        self.assertEqual(
            decode_bundle(encoded, version=3, kind="test"),
            (7, self.RECORDS),
        )

    def test_an_empty_bundle_carries_only_its_rank(self):
        encoded = encode_bundle(1, 2, [])
        self.assertEqual(
            decode_bundle(encoded, version=1, kind="test"),
            (2, ()),
        )

    def test_a_version_mismatch_names_both_versions(self):
        encoded = encode_bundle(1, 0, self.RECORDS)
        with self.assertRaisesRegex(
            ValueError,
            "unsupported Aboleth test-bundle version 1: "
            "this rank speaks version 2",
        ) as raised:
            decode_bundle(encoded, version=2, kind="test")
        self.assertIn(
            "same Aboleth build",
            str(raised.exception),
        )

    def test_a_truncated_bundle_is_rejected(self):
        encoded = encode_bundle(1, 0, self.RECORDS)
        for length in (0, 4, len(encoded) - 1):
            with self.assertRaisesRegex(
                ValueError,
                "truncated Aboleth test",
            ):
                decode_bundle(
                    encoded[:length],
                    version=1,
                    kind="test",
                )

    def test_trailing_bytes_are_rejected(self):
        encoded = encode_bundle(1, 0, self.RECORDS)
        with self.assertRaisesRegex(
            ValueError,
            "trailing bytes in Aboleth test bundle",
        ):
            decode_bundle(
                encoded + b"\0",
                version=1,
                kind="test",
            )


if __name__ == "__main__":
    unittest.main()
