import argparse
import multiprocessing
import os
import traceback
from dataclasses import replace
from pathlib import Path

import torch

from aboleth.runtime import (
    AllocationHandle,
    AllocationProvider,
    RemoteHandle,
    Runtime,
    connection_info_from_bytes,
)
from aboleth_local_endpoint_smoke import (
    TRANSFER_BYTES,
    consumer_kernel,
    producer_kernel,
)


ROOT = Path(__file__).resolve().parents[2]


def _worker(
    device: int,
    peer: int,
    producer: bool,
    transport: str,
    connection,
    barrier,
) -> None:
    os.environ["TRITON_CACHE_DIR"] = str(
        ROOT / ".triton" / "multiprocess-cache" / f"gpu{device}"
    )
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        ROOT / ".torch-cache" / f"gpu{device}"
    )
    try:
        torch.cuda.set_device(device)
        control = torch.zeros(
            16,
            dtype=torch.uint8,
            device=f"cuda:{device}",
        )
        data = torch.zeros(
            TRANSFER_BYTES,
            dtype=torch.uint8,
            device=f"cuda:{device}",
        )
        first_signal = control[:8].view(torch.int64)
        second_signal = control[8:16].view(torch.int64)
        if producer:
            pattern = (
                (
                    torch.arange(
                        TRANSFER_BYTES,
                        device=f"cuda:{device}",
                    )
                    * 17
                    + device
                    + 3
                )
                & 0xFF
            ).to(torch.uint8)
            data.copy_(pattern)
            second_signal.fill_(1)

        with Runtime(device=device) as runtime:
            network = transport != "p2p"
            data_buffer = runtime.register(
                data,
                network=network,
            )
            control_buffer = runtime.register(
                control,
                network=network,
            )
            with runtime.create_local_endpoint(
                transport=transport
            ) as endpoint:
                connection.send(
                    {
                        "connection": (
                            device
                            if transport == "p2p"
                            else endpoint.connection_info.to_bytes()
                        ),
                        "data": (
                            runtime.export_allocation(
                                data_buffer,
                                peer=device,
                                provider=(
                                    AllocationProvider.CUDA_IPC
                                ),
                            ).to_bytes()
                            if transport == "p2p"
                            else runtime.export_buffer(
                                data_buffer,
                                peer=device,
                            ).to_bytes()
                        ),
                        "control": (
                            runtime.export_allocation(
                                control_buffer,
                                peer=device,
                                provider=(
                                    AllocationProvider.CUDA_IPC
                                ),
                            ).to_bytes()
                            if transport == "p2p"
                            else runtime.export_buffer(
                                control_buffer,
                                peer=device,
                            ).to_bytes()
                        ),
                        "hca": (
                            ""
                            if transport == "p2p"
                            else runtime.hca
                        ),
                        "lid": (
                            0
                            if transport == "p2p"
                            else runtime.lid
                        ),
                    }
                )
                remote = connection.recv()
                if transport == "p2p":
                    endpoint.connect_p2p(
                        peer=peer,
                        peer_device=remote["connection"],
                    )
                else:
                    endpoint.connect(
                        connection_info_from_bytes(
                            remote["connection"]
                        ),
                        peer=peer,
                    )
                remote_data_handle = (
                    AllocationHandle.from_bytes(remote["data"])
                    if transport == "p2p"
                    else RemoteHandle.from_bytes(remote["data"])
                )
                remote_control_handle = (
                    AllocationHandle.from_bytes(
                        remote["control"]
                    )
                    if transport == "p2p"
                    else RemoteHandle.from_bytes(
                        remote["control"]
                    )
                )
                remote_data = endpoint.import_remote(
                    remote_data_handle
                )
                remote_control = endpoint.import_remote(
                    remote_control_handle
                )
                if transport == "p2p":
                    try:
                        endpoint.import_remote(
                            replace(
                                remote_data_handle,
                                generation=(
                                    remote_data_handle.generation
                                    + 1
                                ),
                            )
                        )
                    except RuntimeError as error:
                        assert "stale" in str(error)
                    else:
                        raise AssertionError(
                            "stale IPC generation was accepted"
                        )
                barrier.wait(timeout=120)
                with endpoint.compilation():
                    if producer:
                        producer_kernel[(1,)](
                            endpoint.device_handle,
                            remote_data.address,
                            data,
                            remote_control.address,
                            second_signal,
                            endpoint.local_buffer(
                                data_buffer
                            ).lkey,
                            remote_data.rkey,
                            endpoint.local_buffer(
                                control_buffer
                            ).lkey,
                            remote_control.rkey,
                            byte_count=TRANSFER_BYTES,
                            num_warps=1,
                        )
                    else:
                        consumer_kernel[(1,)](
                            endpoint.device_handle,
                            first_signal,
                            second_signal,
                            remote_control.address,
                            endpoint.local_buffer(
                                control_buffer
                            ).lkey,
                            remote_control.rkey,
                            num_warps=1,
                        )
                    torch.cuda.synchronize(device)

                barrier.wait(timeout=120)
                if producer:
                    assert first_signal.cpu().item() == 1
                else:
                    expected = (
                        (
                            torch.arange(
                                TRANSFER_BYTES,
                                device=f"cuda:{device}",
                            )
                            * 17
                            + peer
                            + 3
                        )
                        & 0xFF
                    ).to(torch.uint8)
                    torch.testing.assert_close(
                        data,
                        expected,
                        rtol=0,
                        atol=0,
                    )
                    assert first_signal.cpu().item() == 1

                state = endpoint.state()
                assert state[2] == 0
                remote_data.close()
                remote_control.close()
                if transport == "p2p":
                    try:
                        endpoint.import_remote(
                            replace(
                                remote_data_handle,
                                generation=max(
                                    0,
                                    (
                                        remote_data_handle.generation
                                        - 1
                                    ),
                                ),
                            )
                        )
                    except RuntimeError as error:
                        assert "stale" in str(error)
                    else:
                        raise AssertionError(
                            "post-close stale generation "
                            "was accepted"
                        )
                barrier.wait(timeout=120)
                connection.send(
                    {
                        "ok": True,
                        "device": device,
                        "peer": peer,
                        "hca": (
                            ""
                            if transport == "p2p"
                            else runtime.hca
                        ),
                        "peer_hca": remote["hca"],
                        "lid": (
                            0
                            if transport == "p2p"
                            else runtime.lid
                        ),
                        "peer_lid": remote["lid"],
                        "state": state,
                    }
                )
    except BaseException:
        try:
            connection.send(
                {
                    "ok": False,
                    "device": device,
                    "error": traceback.format_exc(),
                }
            )
        finally:
            raise
    finally:
        connection.close()


def run(transport: str, all_gpus: bool) -> None:
    if all_gpus:
        pairs = ((0, 4), (1, 5), (2, 6), (3, 7))
    else:
        pairs = ((0, 4),)

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2 * len(pairs))
    processes = {}
    parent_connections = {}
    peers = {}
    results = []
    try:
        for left, right in pairs:
            for device, peer, producer in (
                (left, right, True),
                (right, left, False),
            ):
                (
                    parent_connection,
                    child_connection,
                ) = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker,
                    args=(
                        device,
                        peer,
                        producer,
                        transport,
                        child_connection,
                        barrier,
                    ),
                )
                try:
                    process.start()
                except Exception:
                    parent_connection.close()
                    raise
                finally:
                    child_connection.close()
                processes[device] = process
                parent_connections[device] = (
                    parent_connection
                )
                peers[device] = peer

        bundles = {}
        for device, connection in parent_connections.items():
            if not connection.poll(180):
                raise TimeoutError(
                    f"GPU {device} did not publish endpoint metadata"
                )
            bundles[device] = connection.recv()
            if bundles[device].get("ok") is False:
                raise RuntimeError(
                    bundles[device]["error"]
                )

        for device, connection in parent_connections.items():
            connection.send(bundles[peers[device]])

        for device, connection in parent_connections.items():
            if not connection.poll(240):
                raise TimeoutError(
                    f"GPU {device} did not finish its transfer"
                )
            result = connection.recv()
            if not result["ok"]:
                raise RuntimeError(result["error"])
            results.append(result)

        for process in processes.values():
            process.join(timeout=30)
            if process.is_alive():
                raise TimeoutError(
                    f"worker {process.pid} did not exit"
                )
            if process.exitcode != 0:
                raise RuntimeError(
                    f"worker {process.pid} exited with "
                    f"status {process.exitcode}"
                )
    finally:
        try:
            barrier.abort()
        except Exception:
            pass
        for connection in parent_connections.values():
            connection.close()
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        for process in processes.values():
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)

    for result in sorted(
        results,
        key=lambda item: item["device"],
    ):
        print(
            f"gpu{result['device']}/{result['hca']} "
            f"lid={result['lid']:#x} -> "
            f"gpu{result['peer']}/{result['peer_hca']} "
            f"lid={result['peer_lid']:#x} "
            f"state={result['state']}"
        )
    print(
        f"aboleth_multiprocess_{transport}=PASS "
        f"processes={len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("p2p", "proxy", "rdma"),
        required=True,
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
    )
    arguments = parser.parse_args()
    run(arguments.transport, arguments.all_gpus)


if __name__ == "__main__":
    main()
