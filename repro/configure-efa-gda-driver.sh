#!/usr/bin/env bash
set -euo pipefail

CONFIG=/etc/modprobe.d/aboleth-efa-gda.conf

sudo tee "$CONFIG" >/dev/null <<'EOF'
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
EOF

printf 'Installed %s\n' "$CONFIG"
printf 'Reboot the node before running fully device-initiated EFA GDA.\n'
