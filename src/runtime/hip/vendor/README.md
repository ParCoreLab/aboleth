# Ionic direct-verbs declarations

`ionic_dv.h` is from
`ROCm/rocm-systems@2b22ab0195cc1461cd9abf3b969e9dd7c10af350`,
`projects/rocshmem/src/gda/ionic/ionic_dv.h`.
Its original SPDX license and copyright are retained. The only local change
replaces rocSHMEM's internal `gda/ibv_core.hpp` include with the public
`infiniband/verbs.h` include. Aboleth resolves the declared provider functions
with `dlsym`; this does not link or wrap the rocSHMEM library.

Aboleth's own descriptor binding, lowering and device implementation live in
`IonicAbi.h`, `LowerIonic.cpp`, `ionic_control.h` and `ionic_device.cpp`.
