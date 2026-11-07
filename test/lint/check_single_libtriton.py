"""Assert the process maps exactly one libtriton.

glibc identifies a shared object by inode, so a plugin that resolves a *copy* of
libtriton.so rather than the one Python already imported gets its own MLIR
statics. Nothing crashes: types simply stop comparing equal across the plugin
boundary, and a pass silently fails to match ops it built itself. Counting the
distinct libtriton files in /proc/self/maps after importing both is the cheapest
way to notice.
"""

from __future__ import annotations

import pathlib
import re
import sys

import aboleth  # noqa: F401  (configures TRITON_PLUGIN_PATHS before Triton loads)
import triton  # noqa: F401
from triton._C.libtriton import ir  # noqa: F401


def mapped_libtritons() -> set[str]:
    text = pathlib.Path("/proc/self/maps").read_text()
    return {
        str(pathlib.Path(match).resolve())
        for match in re.findall(r"\S*libtriton\.so\S*", text)
    }


def main() -> int:
    mappings = mapped_libtritons()
    for path in sorted(mappings):
        print(path)
    if len(mappings) != 1:
        print(
            f"error: expected exactly one libtriton mapping, found {len(mappings)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
