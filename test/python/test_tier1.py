"""The Tier-1 op programs: lowered shape and AOT artifact compilation.

Everything here runs without a GPU and without a built device library — the
Tier-1 programs compile with detached inline emission (`device_bitcode=None`),
which is the point. The behavioural gate is the hardware WQE differential;
these tests pin the compile-time contract: no library calls, per-thread
reservation, no lane-predicated stores, one doorbell per batch, a strong
`signal` (compare-and-swap, or a fenced write for strictly ordered
registrations), and a stable exported symbol per op.
"""

import pathlib
import re
import tempfile
import unittest

from aboleth import artifact as artifact_module
from aboleth.artifact import compile_fragment_artifact
from aboleth.tier1 import (
    DEFAULT_PUT_MANY_SIZES,
    put_inline_program,
    put_inline_wide_program,
    put_many_flush_program,
    put_many_program,
    put_many_signal_program,
    put_program,
    put_signal_program,
    quiet_program,
    signal_program,
    tier1_programs,
    tier1_symbol,
    wait_signal_program,
)

ARCH = "sm_80"

# fm = 3 in fm_ce_se, the fenced write's control word.
STRONG_ORDER_WORD = 0x60000000
# The responder compare-and-swap's opcode (0x11) or'ed into its WQE's
# index word; a bare constant 17 also appears in descriptor addressing.
COMPARE_SWAP = re.compile(r"arith\.ori %\w+, %c17_i32(_\d+)? ")

# The posting sequence's ordering contract after the fence-downgrade pass:
# exactly one system release (the doorbell), a relaxed DBREC, a GPU-scope
# ready-head release.
DOORBELL = "st.release.sys.global.L1::no_allocate.b64"
DBREC = "st.relaxed.sys.global.L1::no_allocate.b32"
READY = "st.release.gpu.global.L1::no_allocate.b64"


def lowered(program, transport: str = "rdma",
            exclusive: bool = False,
            strict_registrations: bool = False) -> str:
    """The fragment IR after the detached transport lowering."""
    with tempfile.TemporaryDirectory(prefix="aboleth-tier1-") as scratch:
        staging = pathlib.Path(scratch)
        plan = staging / "transport-program.mlir"
        plan.write_text(artifact_module._plan_module(program))
        module = staging / "fragment.mlir"
        module.write_text(
            artifact_module._fragment_module(
                program,
                tier1_symbol("probe", transport),
                artifact_module._slots(program),
            )
        )
        return artifact_module._lower(
            module,
            plan,
            transport=transport,
            completion_interval=1,
            lower_arguments=artifact_module._lower_arguments(
                program, transport, None, exclusive
            ),
            strict_registrations=strict_registrations,
        )


def count(needle: str, haystack: str) -> int:
    return haystack.count(needle)


class Tier1LoweredShapeTest(unittest.TestCase):
    def test_every_op_lowers_without_library_calls_or_lane_guards(self):
        for op, program in tier1_programs().items():
            with self.subTest(op=op):
                text = lowered(program)
                self.assertNotIn('symbol = "aboleth_rdma_', text)
                self.assertNotIn("tt.atomic_rmw", text)
                self.assertNotIn("setp.eq.u32 first, tid, 0", text)
                self.assertNotIn("aboleth.", text)

    def test_put_posting_sequence_has_exactly_one_system_release(self):
        text = lowered(put_program())
        self.assertEqual(
            count("atom.relaxed.gpu.global.add.u64", text), 1
        )
        # The ordering contract: relaxed DBREC, one system-release doorbell,
        # GPU-scope ready head — nothing else at system scope.
        self.assertEqual(count(DBREC, text), 1)
        self.assertEqual(count(DOORBELL, text), 1)
        self.assertEqual(count(READY, text), 1)
        self.assertEqual(count("st.release.sys", text), 1)

    def assertStrongPublication(self, program):
        # Relaxed-ordering registrations: a compare-and-swap. Strictly ordered
        # ones: a fenced write.
        relaxed = lowered(program)
        self.assertRegex(relaxed, COMPARE_SWAP)
        self.assertNotIn(str(STRONG_ORDER_WORD), relaxed)
        strict = lowered(program, strict_registrations=True)
        self.assertIn(str(STRONG_ORDER_WORD), strict)
        self.assertNotRegex(strict, COMPARE_SWAP)

    def test_signal_is_strong_and_put_inline_is_not(self):
        self.assertStrongPublication(signal_program())
        inline = lowered(put_inline_program(), strict_registrations=True)
        self.assertNotIn(str(STRONG_ORDER_WORD), inline)
        self.assertNotRegex(inline, COMPARE_SWAP)

    def test_put_inline_wide_variants_carry_multiword_payloads(self):
        for words in (2, 3):
            with self.subTest(words=words):
                text = lowered(put_inline_wide_program(words))
                self.assertNotIn(str(STRONG_ORDER_WORD), text)
                self.assertEqual(count(DOORBELL, text), 1)

    def test_wait_signal_is_a_bare_acquire_spin(self):
        text = lowered(wait_signal_program())
        self.assertIn("ld.acquire.sys.global", text)
        self.assertNotIn("atom.relaxed", text)
        self.assertNotIn("st.release", text)

    def test_put_many_is_one_reservation_and_one_doorbell(self):
        for size in DEFAULT_PUT_MANY_SIZES:
            with self.subTest(size=size):
                program = put_many_program(size)
                # Four descriptor words per put in the authored body (the
                # lowering adds its own endpoint-word loads on top).
                self.assertEqual(
                    count(" tt.load", program.body), 4 * size
                )
                text = lowered(program)
                self.assertEqual(
                    count("atom.relaxed.gpu.global.add.u64", text), 1
                )
                self.assertEqual(count(DBREC, text), 1)
                self.assertEqual(count(DOORBELL, text), 1)

    def test_put_signal_fuses_the_epoch_into_one_batch(self):
        text = lowered(put_signal_program())
        # One reservation, one doorbell — and a strong trailing epoch.
        self.assertEqual(
            count("atom.relaxed.gpu.global.add.u64", text), 1
        )
        self.assertEqual(count(DOORBELL, text), 1)
        self.assertStrongPublication(put_signal_program())

    def test_put_many_signal_is_still_one_doorbell(self):
        text = lowered(put_many_signal_program(8))
        self.assertEqual(
            count("atom.relaxed.gpu.global.add.u64", text), 1
        )
        self.assertEqual(count(DOORBELL, text), 1)
        self.assertStrongPublication(put_many_signal_program(8))

    def test_put_many_flush_carries_its_own_completion(self):
        text = lowered(put_many_flush_program(8))
        # One reservation, one doorbell, and the CQE poll (its timeout reads
        # %clock64) — where put_many + quiet would pay two of each posting.
        self.assertEqual(
            count("atom.relaxed.gpu.global.add.u64", text), 1
        )
        self.assertEqual(count(DOORBELL, text), 1)
        self.assertIn("%clock64", text)

    def test_exclusive_drops_the_atomic_and_the_order_spin(self):
        plain = lowered(put_program())
        text = lowered(put_program(), exclusive=True)
        self.assertNotIn("atom.relaxed.gpu.global.add.u64", text)
        self.assertEqual(count(DOORBELL, text), 1)
        # The publication-order spin is gone: strictly fewer loops than the
        # concurrent-producer emission (capacity wait stays).
        self.assertLess(count("scf.while", text), count("scf.while", plain))


class Tier1P2PLoweredShapeTest(unittest.TestCase):
    """The p2p program set through the detached P2P lowering.

    No queue exists on peer memory, so the contract is different: every
    store lands from the calling thread (no lane-0 predication, no
    once-per-block tt.atomic_rmw), nothing imposes a block barrier on the
    caller, and quiet is vacuous.
    """

    def test_every_op_lowers_without_barriers_or_lane_guards(self):
        for op, program in tier1_programs("p2p").items():
            with self.subTest(op=op):
                text = lowered(program, "p2p")
                self.assertNotIn("aboleth.", text)
                self.assertNotIn("tt.atomic_rmw", text)
                self.assertNotIn('"bar.sync 0;', text)
                self.assertNotIn("setp.eq.u32 first, tid, 0", text)

    def test_put_is_a_wide_per_thread_copy(self):
        text = lowered(tier1_programs("p2p")["put"], "p2p")
        self.assertIn(
            "{ mov.b64 $0, $2; st.global.L1::no_allocate.b64 [$1], $2; }",
            text,
        )
        # The wide mode never takes the cooperative tensor path.
        self.assertNotIn("tt.make_range", text)

    def test_put_warp_is_a_warp_fenced_cooperative_copy(self):
        text = lowered(tier1_programs("p2p")["put_warp"], "p2p")
        self.assertIn("tt.make_range", text)
        self.assertIn("bar.warp.sync 0xffffffff;", text)

    def test_signal_releases_and_put_inline_does_not(self):
        programs = tier1_programs("p2p")
        signal = lowered(programs["signal"], "p2p")
        inline = lowered(programs["put_inline"], "p2p")
        self.assertIn("st.release.sys.global.b64", signal)
        self.assertIn("st.global.b64", inline)
        self.assertNotIn("st.release", inline)

    def test_quiet_is_vacuous(self):
        text = lowered(tier1_programs("p2p")["quiet"], "p2p")
        for effect in ("st.", "atom.", "tt.store", "tt.load"):
            self.assertNotIn(effect, text)


class Tier1ArtifactTest(unittest.TestCase):
    def _compile_all(self, transport: str, scratch: str):
        for op, program in tier1_programs(transport).items():
            with self.subTest(op=op):
                symbol = tier1_symbol(op, transport)
                artifact = compile_fragment_artifact(
                    program,
                    arch=ARCH,
                    transport=transport,
                    completion_interval=1,
                    device_bitcode=None,
                    output_dir=scratch,
                    symbol=symbol,
                )
                self.assertEqual(artifact.symbol, symbol)
                for path in artifact.files:
                    self.assertTrue(path.is_file(), path)
                text = artifact.ptx.read_text()
                visible = re.findall(
                    r"^\.visible \.func\s+.*?([A-Za-z_][\w$]*)\($",
                    text,
                    re.MULTILINE,
                )
                self.assertEqual(visible, [symbol])
                self.assertNotIn("aboleth_rdma_", text)

    def test_every_rdma_op_compiles_to_its_symbol(self):
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-artifacts-"
        ) as scratch:
            self._compile_all("rdma", scratch)

    def test_every_p2p_op_compiles_to_its_symbol(self):
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-p2p-artifacts-"
        ) as scratch:
            self._compile_all("p2p", scratch)

    def test_signal_and_put_inline_do_not_alias(self):
        # Same slots, different bodies: the signature and the
        # content-addressed stem must separate them.
        with tempfile.TemporaryDirectory(
            prefix="aboleth-tier1-alias-"
        ) as scratch:
            inline = compile_fragment_artifact(
                put_inline_program(),
                arch=ARCH,
                completion_interval=1,
                device_bitcode=None,
                output_dir=scratch,
                symbol=tier1_symbol("put_inline"),
            )
            signal = compile_fragment_artifact(
                signal_program(),
                arch=ARCH,
                completion_interval=1,
                device_bitcode=None,
                output_dir=scratch,
                symbol=tier1_symbol("signal"),
            )
            self.assertNotEqual(
                inline.binding_signature, signal.binding_signature
            )
            self.assertNotEqual(inline.bitcode.name, signal.bitcode.name)

    def test_quiet_reuses_the_planned_drain_program(self):
        program = quiet_program()
        self.assertTrue(program.drain_endpoint)
        self.assertEqual(program.issuer_scope, "thread")


if __name__ == "__main__":
    unittest.main()
