"""dsl-moe arm: DeepEP's dispatch and combine (the paper's EP baseline).

DeepEP v2's `ElasticBuffer` (CUDA kernels over NCCL's device API; an
unmodified checkout named by ABOLETH_DEEPEP_ROOT), at its defaults (automatic SM
and QP counts, hybrid mode), as its tests/elastic/test_ep.py drives it. One
buffer sized for the largest token count; each call passes the cell's count
as `num_max_tokens_per_rank`, its best layout for it, with its default CPU
sync (exact received counts; `ABOLETH_DEEPEP_CPU_SYNC=0` skips it). FP8: DeepEP dispatches
what its caller already quantized, so its own cast
(`deep_ep.utils.math.per_token_cast_to_fp8`, group 128) runs inside the
timed dispatch, as the other arms quantize inside theirs. Its combine sums
the experts' rows; the routing weights are applied on the expert side, as
its tests do.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# DeepEP's FP8 cast is torch.compile'd; Inductor's compile workers do not
# see the Triton driver in this environment, so it compiles in-process.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import moe_common  # noqa: E402

# A DeepEP checkout (v2.5.0) built for this environment.
DEEPEP = os.environ.get("ABOLETH_DEEPEP_ROOT")
if DEEPEP:
    sys.path.insert(0, DEEPEP)
import deep_ep  # noqa: E402
from deep_ep.utils.math import per_token_cast_to_fp8  # noqa: E402


class _Cell:
    def __init__(self, ctx, tokens: int):
        self.ctx, self.tokens = ctx, tokens
        s = ctx.spec
        self.first = s.rank * s.local_experts

    def dispatch(self, x, ids, weights):
        s = self.ctx.spec
        payload = per_token_cast_to_fp8(x) if s.fp8 else x
        recv_x, recv_ids, recv_weights, handle, _ = self.ctx.buffer.dispatch(
            payload, topk_idx=ids.to(deep_ep.topk_idx_t),
            topk_weights=weights, num_experts=s.experts,
            num_max_tokens_per_rank=max(self.tokens, 1),
            expert_alignment=self.ctx.alignment,
            do_cpu_sync=self.ctx.cpu_sync)
        return recv_x, recv_ids, recv_weights, handle

    def counts(self, handle):
        return handle[3].num_unaligned_recv_tokens_per_expert[
            :self.ctx.spec.local_experts]

    def expert(self, handle):
        s = self.ctx.spec
        recv_x, recv_ids, recv_weights, _ = handle
        if isinstance(recv_x, tuple):
            values = moe_common.dequantize(recv_x[0], recv_x[1])
        else:
            values = recv_x.float()
        ids = recv_ids.long()
        # Local or global expert ids: global ones on any rank but 0 are at
        # least E/world.
        if bool((ids >= s.local_experts).any()):
            ids = ids - self.first
        valid = (ids >= 0) & (ids < s.local_experts)
        # Only the rows that arrived (without the CPU sync the buffers are
        # sized for the worst case; DeepEP's device count bounds them).
        received = handle[3].psum_num_recv_tokens_per_scaleup_rank[-1]
        valid &= (torch.arange(ids.shape[0], device=ids.device)
                  < received)[:, None]
        coefficient = torch.where(
            valid, recv_weights * moe_common.expert_factor(ids + self.first),
            0).sum(dim=1)
        return (values * coefficient[:, None]).to(torch.bfloat16)

    def combine(self, handle, y):
        combined, _, _ = self.ctx.buffer.combine(y, handle[3])
        return combined

    def resources(self):
        return []

    def describe(self):
        return {"num_max_tokens_per_rank": max(self.tokens, 1)}

    def close(self):
        torch.cuda.synchronize()


class _Context:
    def __init__(self, args, spec):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        self.spec = spec
        # DeepEP's default synchronizes with the CPU for exact received
        # counts; ABOLETH_DEEPEP_CPU_SYNC=0 skips it (graph-capturable).
        self.cpu_sync = os.environ.get("ABOLETH_DEEPEP_CPU_SYNC", "1") != "0"
        self.alignment = int(os.environ.get("ABOLETH_DEEPEP_ALIGNMENT", "1"))
        self.device = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.device)
        free_before = torch.cuda.mem_get_info()[0]
        self.bootstrap = moe_common.bootstrap_class().from_env(timeout=float(
            os.environ.get("ABOLETH_BOOTSTRAP_TIMEOUT_S", "300")))
        dist.init_process_group(
            "nccl", store=dist.PrefixStore("deepep", self.bootstrap._store),
            rank=spec.rank, world_size=spec.world,
            device_id=torch.device("cuda", self.device))
        most = max(int(t) for t in args.tokens.split(","))
        self.buffer = deep_ep.ElasticBuffer(
            dist.group.WORLD, num_max_tokens_per_rank=max(most, 1),
            hidden=spec.hidden, num_topk=spec.topk,
            use_fp8_dispatch=spec.fp8, explicitly_destroy=True)
        free_after = torch.cuda.mem_get_info()[0]
        # NCCL's communicator (DeepEP's transport) and DeepEP's buffer.
        self.memory = {"free_before": free_before, "free_after": free_after,
                       "library_bytes": free_before - free_after}
        try:
            commit = subprocess.run(
                ["git", "-C", DEEPEP, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        self.meta = {"library": "deepep", "frontend": "cuda",
                     "buffer": "ElasticBuffer", "deepep_commit": commit,
                     "do_cpu_sync": self.cpu_sync,
                     "expert_alignment": self.alignment,
                     "nccl": ".".join(map(str, torch.cuda.nccl.version()))}

    def cell(self, tokens: int) -> _Cell:
        return _Cell(self, tokens)

    def close(self):
        torch.cuda.synchronize()
        self.buffer.destroy()
        dist.destroy_process_group()


def open(args, spec):  # noqa: A001 - the arm protocol's name
    return _Context(args, spec)
