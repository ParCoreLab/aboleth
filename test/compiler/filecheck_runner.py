"""Run one compiler pipeline and check its output with FileCheck.

Replaces the shell one-liners the Makefile used to carry, so each case can be a
separate ctest with its own name, and so the three shapes a case can take —
plain, must-fail, must-be-idempotent — are stated rather than re-spelled.

    filecheck_runner.py --runner run.py --filecheck FileCheck --input case.mlir
                        [--pass NAME]... [--check-prefix P]
                        [--expect-fail] [--idempotent]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--filecheck", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pass", dest="passes", action="append", default=[])
    parser.add_argument("--check-prefix")
    parser.add_argument(
        "--expect-fail",
        action="store_true",
        help="the pipeline must reject the module; its diagnostics are checked",
    )
    parser.add_argument(
        "--idempotent",
        action="store_true",
        help="also require running the pipeline twice to be a fixed point",
    )
    return parser.parse_args()


def run_pipeline(
    args: argparse.Namespace,
    passes: list[str],
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, args.runner, args.input]
    for name in passes:
        command += ["--pass", name]
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    args = parse_args()

    completed = run_pipeline(args, args.passes)
    if args.expect_fail:
        if completed.returncode == 0:
            print(
                f"{args.input}: pipeline succeeded but was expected to fail",
                file=sys.stderr,
            )
            print(completed.stdout, file=sys.stderr)
            return 1
        # A rejection is reported through diagnostics, so those are the subject.
        subject = completed.stdout + completed.stderr
    else:
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            return completed.returncode
        subject = completed.stdout

    check = [args.filecheck, args.input]
    if args.check_prefix:
        check.append(f"--check-prefix={args.check_prefix}")
    checked = subprocess.run(check, input=subject, capture_output=True, text=True)
    if checked.returncode != 0:
        sys.stderr.write(checked.stdout)
        sys.stderr.write(checked.stderr)
        return checked.returncode

    if args.idempotent:
        twice = run_pipeline(args, args.passes + args.passes)
        if twice.returncode != 0:
            print(twice.stderr, file=sys.stderr)
            return twice.returncode
        if twice.stdout != completed.stdout:
            print(
                f"{args.input}: {'+'.join(args.passes)} is not idempotent",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
