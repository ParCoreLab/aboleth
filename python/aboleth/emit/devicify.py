"""Turn a Triton-emitted LLVM module into a linkable NVPTX device library.

Triton only ever emits kernels, so the fragment is compiled as one and these
text transforms rewrite it into a device function:

  1. `ptx_kernel` -> the default calling convention, so the NVPTX backend emits
     a `.visible .func` with an ordinary device-function ABI instead of a
     `.visible .entry`, and the block-size requirement it carried goes;
  2. the scratch pointers Triton appends to every kernel are dropped, and the
     parameter list that remains is checked against the fragment's slots;
  3. debug metadata and module flags go, because they collide with the module
     flags of any clang-compiled CUDA translation unit this links into;
  4. the exported name becomes an entry point around the now-internal body,
     which normalizes three things at once: an i64 result, because Triton's
     extern_elementwise call sites cannot bind a void symbol; a generic pointer
     for any `ptr addrspace(1)` parameter, because that is what a CUDA caller
     hands over; and a thread-0 guard, because a device library call site
     carries none of the lane predication Triton applies in-kernel.

A kernel rather than a `tt.func private` because a kernel's parameter list is
pinned by its entry-point ABI. An internal device function loses its unused
parameters to the optimizer — a completion interval of 1 leaves the iteration
slots dead — which moves the artifact's ABI out from under its shim header and
is caught only by a linker that checks prototypes.

Every step asserts on the shape it expects: a Triton or LLVM change that moves
these lines must fail here rather than silently emit a mismatched ABI.
"""

from __future__ import annotations

import re


_PUBLIC_FUNCTION = re.compile(r"^\s*tt\.func public @([A-Za-z_][\w.$]*)\(",
                              re.MULTILINE)
_MODULE_IDENTITY = re.compile(
    r"^(; ModuleID = ).*$|^(source_filename = ).*$",
    re.MULTILINE,
)


def assert_single_public_entry(module: str) -> None:
    """Zero public functions segfault Triton's IRSource path and two trip an
    assertion inside it, so the count is checked where it can be named."""
    entries = _PUBLIC_FUNCTION.findall(module)
    if len(entries) != 1:
        raise ValueError(
            "a fragment module needs exactly one `tt.func public` entry, "
            f"found {len(entries)}: {', '.join(entries) or 'none'}"
        )


def _find_define(text: str, name: str) -> re.Match:
    pattern = re.compile(
        r"^define\s+[^\n]*@" + re.escape(name) + r"\((?P<args>.*)\)"
        r"(?P<post>[^\n{]*)\{$",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one `define ... @{name}(`, "
            f"found {len(matches)}"
        )
    return matches[0]


def strip_metadata(text: str) -> str:
    """Drop metadata attachments, named metadata, and the nodes they name."""
    # Attachments first (!dbg, !srcloc, !range, ...); their definitions go with
    # the named metadata below.
    text = re.sub(r",?\s*![A-Za-z_][\w.]* !\d+", "", text)
    text = re.sub(r"^!(?!\d)[\w.$]+ = .*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^!\d+ = .*$", "", text, flags=re.MULTILINE)
    text = _MODULE_IDENTITY.sub(
        lambda match: (match.group(1) or match.group(2)) + '"aboleth"',
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text)


def devicify_kernel(text: str, symbol: str, parameters: int) -> str:
    """Rewrite the kernel @symbol into an extern-linkage device function."""
    match = _find_define(text, symbol)
    if "ptx_kernel" not in match.group(0):
        raise ValueError(
            f"@{symbol} does not carry the ptx_kernel calling convention; "
            "the fragment is expected to be compiled as a kernel"
        )
    arguments = [
        argument.strip()
        for argument in match.group("args").split(",")
    ]
    # Triton appends a global and a profile scratch pointer to every kernel;
    # neither is read when global_scratch_size is zero.
    while (
        len(arguments) > parameters and
        "readnone captures(none)" in arguments[-1]
    ):
        arguments.pop()
    if len(arguments) != parameters:
        raise ValueError(
            f"@{symbol} has {len(arguments)} parameters after dropping "
            f"scratch, expected the fragment's {parameters}"
        )

    post = match.group("post")
    group = re.search(r"#(\d+)", post)
    post = " ".join(post.replace("local_unnamed_addr", "").split())
    text = (
        text[:match.start()] +
        f"define void @{symbol}({', '.join(arguments)}) {post} {{" +
        text[match.end():]
    )
    if group is not None:
        # A device function has no block shape to require.
        text = re.sub(
            r"(^attributes #" + group.group(1) + r" = \{[^}]*?)"
            r'\s*"nvvm\.reqntid"="\d+"',
            r"\1",
            text,
            flags=re.MULTILINE,
        )
    return strip_metadata(text)


_TID = "llvm.nvvm.read.ptx.sreg.tid.x"


def wrap_entry_point(
    text: str,
    symbol: str,
    *,
    issuer_scope: str = "cta",
) -> str:
    """Rename @symbol to @symbol_body and add the linkable entry point.

    The entry point
      * returns i64 (always zero), because Triton binds extern device symbols
        through extern_elementwise, which has no void result;
      * casts any global-space pointer parameter from the generic pointer a
        CUDA caller hands over;
      * runs the body on thread 0 for CTA-scoped fragments, or once per caller
        for thread-scoped integration hooks.
    """
    match = _find_define(text, symbol)
    if not match.group(0).startswith("define void "):
        raise ValueError(
            f"@{symbol} must return void before it is wrapped, got "
            f"{match.group(0).split('@')[0].strip()}"
        )
    group = re.search(r"#(\d+)", match.group("post"))
    if group is None:
        raise ValueError(f"@{symbol} carries no attribute group to reuse")

    # The body stays internal so the artifact exports exactly one symbol.
    body = f"{symbol}_body"
    text = (
        text[:match.start()] +
        match.group(0)
        .replace(f"@{symbol}(", f"@{body}(")
        .replace("define void ", "define internal void ") +
        text[match.end():]
    )

    casts = []
    parameters = []
    arguments = []
    for index, argument in enumerate(match.group("args").split(",")):
        argument_type = re.sub(r"\s*%[\w.]+$", "", argument.strip())
        if "addrspace(1)" in argument_type:
            generic = re.sub(r"\s*addrspace\(1\)", "", argument_type)
            parameters.append(f"{generic} %a{index}")
            casts.append(
                f"  %g{index} = addrspacecast {generic} %a{index} to "
                "ptr addrspace(1)"
            )
            arguments.append(f"ptr addrspace(1) %g{index}")
            continue
        parameters.append(f"{argument_type} %a{index}")
        arguments.append(f"{argument_type} %a{index}")

    if issuer_scope not in ("cta", "thread"):
        raise ValueError(
            f"unknown fragment issuer scope {issuer_scope!r}"
        )
    declaration = () if (
        issuer_scope == "thread" or f"@{_TID}(" in text
    ) else (
        f"declare i32 @{_TID}()",
        "",
    )
    if issuer_scope == "thread":
        body_lines = (
            "; Linkable device entry point: every caller performs one "
            "transport operation.",
            f"define i64 @{symbol}({', '.join(parameters)}) "
            f"#{group.group(1)} {{",
            *casts,
            f"  call void @{body}({', '.join(arguments)})",
            "  ret i64 0",
            "}",
        )
    else:
        body_lines = (
            "; Linkable device entry point: every thread of the calling block "
            "may",
            "; call it, thread 0 performs the transport, the i64 result is zero.",
            f"define i64 @{symbol}({', '.join(parameters)}) "
            f"#{group.group(1)} {{",
            f"  %tid = call i32 @{_TID}()",
            "  %first = icmp eq i32 %tid, 0",
            "  br i1 %first, label %post, label %done",
            "post:",
            *casts,
            f"  call void @{body}({', '.join(arguments)})",
            "  br label %done",
            "done:",
            "  ret i64 0",
            "}",
        )
    return "\n".join((
        text.rstrip("\n"),
        "",
        *declaration,
        *body_lines,
        "",
    ))
