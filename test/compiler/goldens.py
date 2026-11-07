"""Capture and compare lowered-IR goldens across every backend.

Backend restructuring must not change the code any lowering emits. This tool
runs every test module through every pipeline that accepts it, including the
environment-selected lowering variants, and records the exact printed module or
the exact diagnostic. `--check` fails on the first byte of drift.

    python test/compiler/goldens.py --write .goldens
    python test/compiler/goldens.py --check .goldens
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pathlib
import sys

import aboleth  # noqa: F401  (installs the dialect and pass registrations)
from triton._C.libtriton import ir, passes


def pass_builder(name: str):
    """`form-batches` names `passes.plugin.add_aboleth_form_batches`."""
    return getattr(passes.plugin, "add_aboleth_" + name.replace("-", "_"))

BACKENDS = ("p2p", "proxy", "rdma", "efa")

# The EFA lowering emits calls into a device library; only the path is baked
# into the IR, so goldens use a fixed placeholder rather than a built artifact.
EFA_BITCODE = "/aboleth/efa.bc"

# Environment knobs that select a different emission inside one backend pass.
BACKEND_VARIANTS = {
    "p2p": (
        {},
        {"ABOLETH_P2P_SCALAR": "1"},
        {"ABOLETH_P2P_COOP": "1"},
        {"ABOLETH_P2P_COOP_MIN": "64"},
    ),
    "proxy": ({},),
    "rdma": ({},),
    "efa": (
        {"ABOLETH_EFA_BITCODE": EFA_BITCODE},
    ),
}

# Pipeline prefixes worth exercising ahead of each backend lowering. Modules
# that reject a prefix simply record the diagnostic, which is itself a golden.
PREFIXES = (
    ("batches", ["form-batches"]),
    ("published", ["plan-publication", "form-batches"]),
    (
        "completion8",
        ["materialize-completion=8", "form-batches"],
    ),
    (
        "dataflow",
        ["simplify-communication-dataflow", "form-batches"],
    ),
)

# Backend-free pipelines: planning and analysis output must stay stable too.
ANALYSIS_PIPELINES = (
    ("parse", []),
    ("plan-transport", ["plan-transport"]),
    ("storage", ["plan-storage"]),
    ("storage-generic", ["plan-storage=,unoptimized"]),
    ("valid-extents", ["materialize-valid-extents"]),
    ("valid-extents-generic", ["materialize-valid-extents=unoptimized"]),
    ("plan-publication", ["plan-publication"]),
    ("simplify", ["simplify-communication-dataflow"]),
    ("form-batches", ["form-batches"]),
    ("materialize1", ["materialize-completion=1"]),
    ("materialize8", ["materialize-completion=8"]),
)


def run_pipeline(module_path: str, pipeline: list[str]) -> str:
    """Return the printed module, or the diagnostic if the pipeline rejects it."""
    context = ir.context()
    ir.load_dialects(context)
    captured = io.StringIO()
    try:
        # MLIR diagnostics reach stderr; keep them in the golden.
        with contextlib.redirect_stderr(captured):
            module = ir.parse_mlir_module(module_path, context)
            if pipeline:
                manager = ir.pass_manager(context)
                for specification in pipeline:
                    name, separator, value = specification.partition("=")
                    arguments = value.split(",") if separator else []
                    pass_builder(name)(manager, arguments)
                manager.run(module, "aboleth-goldens")
            return module.str_nodebug()
    except Exception as error:  # noqa: BLE001 - the message is the golden
        return f"!! rejected\n{captured.getvalue()}{error}\n"


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")


def scenarios(test_dir: pathlib.Path):
    """Yield (name, module_path, pipeline, env) for everything worth pinning."""
    modules = sorted(test_dir.glob("*.mlir"))
    for module in modules:
        stem = module.stem
        analysis = ANALYSIS_PIPELINES
        if "aboleth.endpoint_work" in module.read_text():
            analysis += (("endpoints", ["plan-endpoints"]),
                         ("endpoints-generic", ["plan-endpoints=,unoptimized"]))
        if "aboleth.shape_family" in module.read_text():
            analysis += (("shapes", ["plan-shapes"]),
                         ("shapes-generic", ["plan-shapes=,unoptimized"]))
        if "bindableOutputs = true" in module.read_text():
            analysis += (("binding-generic", ["simplify-communication-dataflow=unoptimized"]),)
        for label, pipeline in analysis:
            yield f"{stem}.{label}", str(module), list(pipeline), {}
        prefixes = PREFIXES
        if "aboleth.bind_valid_local_region" in module.read_text():
            prefixes += (
                ("valid", ["materialize-valid-extents", "plan-publication", "form-batches"]),
                ("valid-generic", ["materialize-valid-extents=unoptimized", "plan-publication", "form-batches"]),
            )
        for prefix_label, prefix in prefixes:
            for backend in BACKENDS:
                for variant in BACKEND_VARIANTS[backend]:
                    # Variant values reach the filename, so keep them to
                    # characters a path accepts.
                    suffix = "".join(
                        f".{key.lower()}-{_slug(value)}"
                        for key, value in sorted(variant.items())
                    )
                    name = f"{stem}.{prefix_label}.{backend}{suffix}"
                    pipeline = list(prefix) + [f"lower-{backend}"]
                    yield name, str(module), pipeline, dict(variant)


def capture(test_dir: pathlib.Path) -> dict[str, str]:
    results: dict[str, str] = {}
    saved = dict(os.environ)
    for name, module_path, pipeline, env in scenarios(test_dir):
        for key in ("ABOLETH_P2P_SCALAR", "ABOLETH_P2P_COOP",
                    "ABOLETH_P2P_COOP_MIN",
                    "ABOLETH_EFA_BITCODE", "ABOLETH_EFA_PUBLICATION"):
            os.environ.pop(key, None)
        os.environ.update(env)
        results[name] = run_pipeline(module_path, pipeline)
    os.environ.clear()
    os.environ.update(saved)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", metavar="DIR")
    parser.add_argument("--check", metavar="DIR")
    parser.add_argument(
        "--test-dir",
        default=str(pathlib.Path(__file__).parent),
    )
    args = parser.parse_args()
    if bool(args.write) == bool(args.check):
        parser.error("pass exactly one of --write or --check")

    test_dir = pathlib.Path(args.test_dir)
    results = capture(test_dir)

    if args.write:
        target = pathlib.Path(args.write)
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.glob("*.txt"):
            stale.unlink()
        for name, text in results.items():
            (target / f"{name}.txt").write_text(text)
        rejected = sum(1 for text in results.values() if text.startswith("!!"))
        print(
            f"wrote {len(results)} goldens to {target} "
            f"({len(results) - rejected} lowered, {rejected} rejected)"
        )
        return 0

    source = pathlib.Path(args.check)
    expected = {
        path.stem: path.read_text() for path in sorted(source.glob("*.txt"))
    }
    drift = []
    for name in sorted(set(expected) | set(results)):
        if name not in expected:
            drift.append(f"new scenario: {name}")
        elif name not in results:
            drift.append(f"missing scenario: {name}")
        elif expected[name] != results[name]:
            drift.append(f"changed output: {name}")
    if drift:
        print(f"{len(drift)} golden differences:", file=sys.stderr)
        for entry in drift[:40]:
            print(f"  {entry}", file=sys.stderr)
        if len(drift) > 40:
            print(f"  ... and {len(drift) - 40} more", file=sys.stderr)
        return 1
    print(f"{len(results)} goldens match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
