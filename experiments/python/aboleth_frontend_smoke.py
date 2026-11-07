import os
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
import aboleth.language as aboleth
from aboleth.compiler import compilation
from triton._C.libtriton import ir, passes
from triton._filecheck import run_parser


@triton.jit
def producer_kernel(
    endpoint,
    destination,
    source,
    remote_signal,
    local_signal,
    byte_count,
):
    transfer = None
    for index in tl.static_range(0, 32):
        offset = index * byte_count
        transfer = aboleth.put(
            endpoint,
            destination + offset,
            source + offset,
            byte_count,
            after=transfer,
        )
    ready = aboleth.signal(
        endpoint,
        remote_signal,
        local_signal,
        transfer,
    )
    aboleth.wait(ready)


@triton.jit
def consumer_kernel(local_signal, expected):
    aboleth.wait_signal(local_signal, expected)


def main() -> None:
    endpoint = torch.empty(12, dtype=torch.int64, device="cuda")
    destination = torch.empty(32 * 256, dtype=torch.uint8, device="cuda")
    source = torch.empty_like(destination)
    remote_signal = torch.zeros(1, dtype=torch.int64, device="cuda")
    local_signal = torch.ones_like(remote_signal)

    producer_module = run_parser(
        producer_kernel,
        args=(
            endpoint,
            destination,
            source,
            remote_signal,
            local_signal,
            256,
        ),
    )
    consumer_module = run_parser(
        consumer_kernel,
        args=(remote_signal, 1),
    )
    producer_ttir = producer_module.str_nodebug()
    consumer_ttir = consumer_module.str_nodebug()
    if os.environ.get("ABOLETH_DUMP_IR") == "1":
        print("=== producer ===")
        print(producer_ttir)
        print("=== consumer ===")
        print(consumer_ttir)

    assert producer_ttir.count("aboleth.put") == 32
    assert producer_ttir.count("tt.addptr") >= 32
    assert "aboleth.signal" in producer_ttir
    assert "aboleth.wait" in producer_ttir
    assert "!aboleth.token" in producer_ttir
    assert "aboleth.bind_endpoint" in producer_ttir
    assert "aboleth.bind_local_region" in producer_ttir
    assert "aboleth.bind_remote_region" in producer_ttir
    assert "!aboleth.endpoint" in producer_ttir
    assert "!aboleth.local_region" in producer_ttir
    assert "!aboleth.remote_region" in producer_ttir
    assert "aboleth.wait_signal" in consumer_ttir

    with tempfile.TemporaryDirectory() as directory:
        for name, module_text in (
            ("producer", producer_ttir),
            ("consumer", consumer_ttir),
        ):
            ir_path = Path(directory) / f"{name}.mlir"
            ir_path.write_text(module_text)
            context = ir.context()
            ir.load_dialects(context)
            parsed = ir.parse_mlir_module(str(ir_path), context)
            assert "aboleth." in parsed.str_nodebug()

        producer_path = Path(directory) / "producer.mlir"
        producer_path.write_text(producer_ttir)
        context = ir.context()
        ir.load_dialects(context)
        scheduled_module = ir.parse_mlir_module(
            str(producer_path),
            context,
        )
        manager = ir.pass_manager(context)
        passes.plugin.add_aboleth_form_batches(manager)
        manager.run(scheduled_module, "aboleth-frontend-smoke")
        scheduled_ttir = scheduled_module.str_nodebug()

    if os.environ.get("ABOLETH_DUMP_IR") == "1":
        print("=== scheduled producer ===")
        print(scheduled_ttir)
    assert scheduled_ttir.count("aboleth.post_put") == 32
    assert scheduled_ttir.count("aboleth.post_signal") == 1
    assert scheduled_ttir.count("aboleth.submit") == 1
    assert scheduled_ttir.count("aboleth.poll") == 1
    assert "batchSize = 33" in scheduled_ttir
    assert "aboleth.put " not in scheduled_ttir
    assert "aboleth.signal " not in scheduled_ttir
    assert "aboleth.wait " not in scheduled_ttir
    assert "source_key" not in scheduled_ttir
    assert "destination_key" not in scheduled_ttir

    if os.environ.get("ABOLETH_RDMA_DEVICE_LIBRARY") == "1":
        with compilation("rdma"):
            compiled = producer_kernel.warmup(
                endpoint,
                destination,
                source,
                remote_signal,
                local_signal,
                256,
                grid=(1,),
                num_warps=1,
            )
            compiled._init_handles()
        ptx = compiled.asm.get("ptx", "")
        residual = [
            line
            for line in ptx.splitlines()
            if (
                "aboleth_rdma_" in line and
                "call" in line
            )
        ]
        assert not residual, residual
        assert "st.release.sys.global" in ptx
        print("rdma_device_library=PASS")

    print("producer_puts=32")
    print("scheduled_batch_size=33")
    print("aboleth_frontend=PASS")


if __name__ == "__main__":
    main()
