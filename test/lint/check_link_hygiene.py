"""Assert what a shared object is and is not allowed to depend on.

    check_link_hygiene.py <file> [--forbid REGEX] [--allow-only REGEX]

The two rules this enforces:

* libaboleth_rdma_abi.so must carry no transport DT_NEEDED. Every provider is
  resolved with dlopen so one binary loads on a machine that has none of them;
  a stray link-time reference would undo that silently.
* libAbolethPlugin.so must need nothing but libtriton and the C++ runtime. MLIR
  lives inside libtriton, and a second copy reached through any other library
  would give the process two sets of TypeIDs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def needed(path: str) -> list[str]:
    output = subprocess.run(
        ["readelf", "-d", path], capture_output=True, text=True, check=True
    ).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--forbid", default=None)
    parser.add_argument("--allow-only", default=None)
    args = parser.parse_args()

    libraries = needed(args.path)
    print(f"{args.path} NEEDED: {' '.join(libraries) or '(none)'}")

    failures = []
    if args.forbid:
        pattern = re.compile(args.forbid)
        failures += [
            f"forbidden dependency {name}"
            for name in libraries
            if pattern.search(name)
        ]
    if args.allow_only:
        pattern = re.compile(args.allow_only)
        failures += [
            f"unexpected dependency {name}"
            for name in libraries
            if not pattern.search(name)
        ]

    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
