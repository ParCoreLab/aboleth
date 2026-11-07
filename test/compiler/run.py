from __future__ import annotations

import argparse
import sys

import aboleth
from triton._C.libtriton import ir, passes


def pass_builder(name: str):
    """`--pass form-batches` adds `passes.plugin.add_aboleth_form_batches`."""
    return getattr(passes.plugin, "add_aboleth_" + name.replace("-", "_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    parser.add_argument("--pass", dest="passes", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(args.module, context)
    if args.passes:
        manager = ir.pass_manager(context)
        for specification in args.passes:
            name, separator, value = specification.partition("=")
            arguments = value.split(",") if separator else []
            pass_builder(name)(manager, arguments)
        manager.run(module, "aboleth-test")
    print(module.str_nodebug())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
