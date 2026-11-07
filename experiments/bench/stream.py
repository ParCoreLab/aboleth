"""Microbenchmark: K independent put+signal chains to one peer per round.

One program on one endpoint issues, every round, K chains of one put and one
strong signal into K separate flag cells. The modes differ only in where the
chains' waits sit and how the chains relate:

  chains  every wait at the end of the round, so the K submissions are
          independent and adjacent: what submission coalescing may merge;
  serial  each chain waited before the next is issued: K full posting and
          completion sequences, nothing to merge;
  fused   one chain of K puts closed by a single signal: the hand-merged
          reference (one flag instead of K, so not the same program);
  pipelined  like chains, but each wait declares that its sources are reused
          only --reuse rounds later, so completion is planned every k-th
          round and the endpoint is drained after the loop;
  empty   the round loop and its timers with no communication: the floor
          under every other mode's issue time.

Each line of output is one JSON record with the median time per round and
per chain, the kernel's registers, spills and cubin size, and provenance.
Every flag must end at the round count and every payload must arrive.
ABOLETH_PASSES selects the optimizations (ablation.py sweeps it).
"""

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import time

import torch

from aboleth.runtime import Runtime, connection_info_from_bytes

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

_HEADER = '''
import triton
import triton.language as tl

import aboleth.language as aboleth


@triton.jit(do_not_specialize=["rounds"])
def kernel(endpoint, remote_address, source, source_key, destination_key,
           elapsed_ns, rounds, BYTES: tl.constexpr):
{select}    payload = remote_address.to(tl.pointer_type(tl.int8))
    flags = (remote_address + {count} * BYTES).to(tl.pointer_type(tl.int64))
    start = tl.extra.cuda.globaltimer()
    issue = tl.zeros([], dtype=tl.int64)
    for round_index in tl.range(0, rounds):
        epoch = (round_index + 1).to(tl.int64)
        issued_from = tl.extra.cuda.globaltimer()
'''

_FOOTER = '''{drain}
    end = tl.extra.cuda.globaltimer()
    tl.store(elapsed_ns, end - start)
    tl.store(elapsed_ns + 1, issue)
'''

# Issue time: from the round's start to the end of its last submission, which
# the compiler places at the last chain's signal, before any wait.
_ISSUED = (
    "        issue += (tl.extra.cuda.globaltimer() - issued_from)"
    ".to(tl.int64)\n"
)


def _put(index: int, after: str | None) -> str:
    chained = f", after={after}" if after else ""
    return (
        f"        t{index} = aboleth.put(endpoint, payload + {index} * BYTES,"
        f" source + {index} * BYTES, BYTES, source_key=source_key,"
        f" destination_key=destination_key{chained})\n"
    )


def _signal(index: int, after: str) -> str:
    return (
        f"        s{index} = aboleth.signal(endpoint, flags + {index}, epoch,"
        f" after={after}, destination_key=destination_key)\n"
    )


# With --exclusive the kernel receives a one-entry endpoint table and selects
# by program id, so the compiler proves a single issuer (plain-add
# reservation, no reservation-order wait), as Tier-2 kernels using
# aboleth.endpoint do.
_SELECT = "    endpoint = aboleth.endpoint(endpoint, tl.program_id(0))\n"


def kernel_source(mode: str, count: int, reuse: int,
                  exclusive: bool = False) -> str:
    body = ""
    drain = ""
    if mode == "empty":
        body = _ISSUED
    elif mode == "pipelined":
        for index in range(count):
            body += _put(index, None) + _signal(index, f"t{index}")
        body += _ISSUED + "".join(
            f"        aboleth.wait(s{index}, iteration=round_index,"
            f" reuse={reuse})\n" for index in range(count))
        drain = "\n    aboleth.drain(endpoint)"
    elif mode == "fused":
        for index in range(count):
            body += _put(index, f"t{index - 1}" if index else None)
        body += _signal(0, f"t{count - 1}")
        body += _ISSUED + "        aboleth.wait(s0)\n"
    else:
        for index in range(count):
            body += _put(index, None) + _signal(index, f"t{index}")
            if mode == "serial":
                body += f"        aboleth.wait(s{index})\n"
        if mode == "chains":
            body += _ISSUED + "".join(
                f"        aboleth.wait(s{index})\n" for index in range(count))
    select = _SELECT if exclusive else ""
    return (_HEADER.format(count=count, select=select) + body
            + _FOOTER.format(drain=drain))


def load_kernel(directory: pathlib.Path, mode: str, count: int, reuse: int,
                exclusive: bool = False):
    path = directory / f"coalesce_{mode}_{count}.py"
    path.write_text(kernel_source(mode, count, reuse, exclusive))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def connection_info(endpoint):
    return connection_info_from_bytes(endpoint.connection_info.to_bytes())


def measure(args, kernel, mode, count, forward, remote_address,
            source_key, destination_key, source, remote):
    elapsed = torch.zeros(2, dtype=torch.int64, device=f"cuda:{args.left}")
    payload_bytes = count * args.bytes
    flags = remote[payload_bytes:payload_bytes + count * 8].view(torch.int64)
    signalled = {"fused": 1, "empty": 0}.get(mode, count)
    # A one-entry endpoint table; the RDMA kind tag is zero.
    endpoint = (torch.tensor([forward.device_handle.data_ptr()],
                             dtype=torch.uint64, device=f"cuda:{args.left}")
                if args.exclusive else forward.device_handle)
    payload_bytes = 0 if mode == "empty" else payload_bytes
    samples = []
    kernel_record = None
    for repeat in range(args.repeats + 1):
        # Epochs restart at one every launch, and a CAS signal replaces only
        # the epoch it last published, so every cell restarts at zero.
        remote.zero_()
        torch.cuda.synchronize(args.right)
        with torch.cuda.device(args.left):
            started = time.perf_counter()
            compiled = kernel[(1,)](
                endpoint, remote_address, source, source_key,
                destination_key, elapsed, args.rounds, BYTES=args.bytes,
                num_warps=1)
            torch.cuda.synchronize(args.left)
            if kernel_record is None:
                kernel_record = {
                    "first_call_s": round(time.perf_counter() - started, 3),
                    "registers": compiled.n_regs,
                    "spills": compiled.n_spills,
                    "cubin_bytes": len(compiled.asm.get("cubin", b""))}
        assert (flags[:signalled].cpu() == args.rounds).all(), (mode, count)
        torch.testing.assert_close(
            remote[:payload_bytes].cpu(), source[:payload_bytes].cpu(),
            rtol=0, atol=0)
        # The first launch also compiles; it is not a sample.
        if repeat:
            total, issue = elapsed.tolist()
            samples.append((total / args.rounds, issue / args.rounds))
    samples.sort()
    median, issue = samples[len(samples) // 2]
    samples = [total for total, _ in samples]
    return {
        "mode": mode,
        "chains": count,
        "bytes": args.bytes,
        "rounds": args.rounds,
        "reuse": args.reuse if mode == "pipelined" else None,
        "ns_per_round": round(median, 1),
        "ns_per_chain": round(median / count, 1),
        # Serial mode waits inside the round, so its issue time is not timed.
        "issue_ns_per_round": None if mode == "serial" else round(issue, 1),
        "ns_per_round_min": round(samples[0], 1),
        "ns_per_round_max": round(samples[-1], 1),
        "label": args.label,
        "exclusive": args.exclusive,
        "gpu_resident_queues": forward.gpu_resident_queues,
        "submit_mode": forward.submit_mode,
        "kernel": kernel_record,
        "provenance": common.provenance(),
    }


def run(args):
    largest = max(args.chains)
    source = (torch.arange(largest * args.bytes, device=f"cuda:{args.left}")
              % 251).to(torch.uint8)
    remote = torch.zeros(largest * (args.bytes + 8), dtype=torch.uint8,
                         device=f"cuda:{args.right}")
    with (
        Runtime(device=args.left) as left_runtime,
        Runtime(device=args.right) as right_runtime,
    ):
        source_buffer = left_runtime.register(source)
        remote_buffer = right_runtime.register(remote)
        if args.fenced and not remote_buffer.strictly_ordered:
            raise SystemExit(
                "--fenced needs a strictly ordered registration: "
                "set ABOLETH_RELAXED_ORDERING=0")
        with (
            left_runtime.create_local_endpoint(
                transport="rdma", qp_depth=args.qp_depth) as forward,
            right_runtime.create_local_endpoint(
                transport="rdma", qp_depth=args.qp_depth) as reverse,
        ):
            forward.connect(connection_info(reverse), peer=args.right)
            reverse.connect(connection_info(forward), peer=args.left)
            imported = forward.import_remote(
                right_runtime.export_buffer(remote_buffer, peer=args.right))
            source_key = forward.local_buffer(source_buffer).lkey
            with (
                tempfile.TemporaryDirectory(prefix="aboleth-coalesce-") as scratch,
                forward.compilation(
                    queue_capacity=forward.capacity,
                    strict_registrations=args.fenced),
            ):
                for mode in args.modes:
                    for count in args.chains:
                        kernel = load_kernel(
                            pathlib.Path(scratch), mode, count, args.reuse,
                            args.exclusive)
                        record = measure(
                            args, kernel, mode, count, forward,
                            imported.address, source_key, imported.rkey,
                            source, remote)
                        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+",
                        default=["serial", "chains", "fused"],
                        choices=["serial", "chains", "fused", "pipelined",
                                 "empty"])
    parser.add_argument("--chains", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--bytes", type=int, default=64)
    parser.add_argument("--reuse", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--qp-depth", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--exclusive", action="store_true",
                        help="select the endpoint from a table by program "
                        "id, so the compiler proves a single issuer")
    parser.add_argument("--fenced", action="store_true",
                        help="strong signals as fenced writes (strictly "
                        "ordered registration) instead of compare-and-swap")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
