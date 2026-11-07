# Vendored mlx5 PRM mailbox layouts

`mlx5_ifc.h` (Mellanox, GPL-2.0 OR Linux-OpenIB dual license) and
`mlx5_prm.h` (BSD-3-Clause) are vendored verbatim from the NVSHMEM source
tree (`src/modules/transport/common/`), which itself carries them from
rdma-core / DPDK. They describe the ConnectX PRM command mailboxes
(CREATE_CQ, CREATE_QP, RST2INIT/INIT2RTR/RTR2RTS) the DEVX-backed staged
endpoint issues to place its send/completion queues and doorbell records in
GPU memory. Never re-type PRM layouts by hand; refresh by re-copying.
