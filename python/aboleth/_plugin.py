from __future__ import annotations

import os
import sys
from pathlib import Path


_hip_libraries = []


def _isolate_hip_compiler(path: Path) -> None:
    # ROCm SDK wheels also carry LLVM. Plugin-enabled Triton exports LLVM
    # symbols, so each shared object's relocations must bind to its own LLVM
    # dependency instead of whichever compiler was imported first.
    if _hip_libraries or not hasattr(os, "RTLD_DEEPBIND"):
        return
    from importlib.metadata import version
    loaded_torch = sys.modules.get("torch")
    if loaded_torch is None and "rocm" not in version("torch").lower():
        return
    import torch
    if not torch.version.hip:
        return
    import ctypes
    import importlib.util
    spec = importlib.util.find_spec("triton")
    library = Path(spec.origin).parent / "_C" / "libtriton.so"
    mode = os.RTLD_NOW | os.RTLD_DEEPBIND
    triton_library = ctypes.CDLL(str(library), mode=mode)
    plugin_library = ctypes.CDLL(str(path), mode=mode)
    _hip_libraries.extend((triton_library, plugin_library))


def plugin_path() -> Path:
    override = os.environ.get("ABOLETH_PLUGIN_PATH")
    if override:
        return Path(override).expanduser().resolve()
    packaged = Path(__file__).resolve().parent / "libAbolethPlugin.so"
    if packaged.is_file():
        return packaged
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("ABOLETH_BUILD_DIR", root / "build"))
    return build / "lib" / "libAbolethPlugin.so"


def configure_plugin() -> Path:
    # Resolved, as the registered paths are compared resolved: a symlinked
    # plugin or build directory must not look like a second plugin.
    path = plugin_path().resolve()
    if not path.is_file():
        raise ImportError(
            f"Aboleth compiler plugin is not built: {path}. "
            "Run `cmake --build build --target AbolethPlugin`."
        )

    existing = [
        Path(item).resolve()
        for item in os.environ.get("TRITON_PLUGIN_PATHS", "").split(":")
        if item
    ]
    if path not in existing:
        if "triton._C.libtriton" in sys.modules:
            raise ImportError(
                "Aboleth must configure its compiler plugin before "
                "triton._C.libtriton is imported"
            )
        existing.append(path)
        os.environ["TRITON_PLUGIN_PATHS"] = ":".join(map(str, existing))
    _isolate_hip_compiler(path)
    return path
