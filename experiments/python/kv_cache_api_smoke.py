from aboleth.frontends.triton.kv_cache import (
    migrate_kv_cache_pages,
)
import torch


def main():
    world_size = 2
    max_pages = 8
    page_bytes = 16 * 1024
    pages = []
    source_rows = []
    destination_peers = []
    destination_rows = []
    counts = []
    for rank in range(world_size):
        values = torch.zeros(
            max_pages,
            page_bytes,
            dtype=torch.uint8,
            device=f"cuda:{rank}",
        )
        values[0].fill_(rank + 1)
        pages.append(values)
        source_rows.append(torch.zeros(
            max_pages,
            dtype=torch.int64,
            device=f"cuda:{rank}",
        ))
        peers = torch.zeros(
            max_pages,
            dtype=torch.int32,
            device=f"cuda:{rank}",
        )
        peers[0] = 1 - rank
        destination_peers.append(peers)
        rows = torch.zeros(
            max_pages,
            dtype=torch.int64,
            device=f"cuda:{rank}",
        )
        rows[0] = 1
        destination_rows.append(rows)
        counts.append(torch.tensor(
            [1],
            dtype=torch.int32,
            device=f"cuda:{rank}",
        ))

    result = migrate_kv_cache_pages(
        tuple(pages),
        tuple(source_rows),
        tuple(destination_peers),
        tuple(destination_rows),
        tuple(counts),
        transport="p2p",
    )
    for rank, actual in enumerate(result.pages):
        expected = torch.full(
            (page_bytes,),
            2 - rank,
            dtype=torch.uint8,
            device=f"cuda:{rank}",
        )
        torch.testing.assert_close(
            actual[1],
            expected,
            rtol=0,
            atol=0,
        )
    print("kv_cache_api=PASS")


if __name__ == "__main__":
    main()
