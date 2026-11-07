# Drops the !llvm.module.flags line from a textual LLVM module. Run as
# `cmake -DINPUT=... -DOUTPUT=... -P StripModuleFlags.cmake` so the bitcode rule
# does not need a shell.
#
# Read as one string rather than with file(STRINGS): LLVM IR is full of `;`
# comments, which file(STRINGS) would turn into list separators.
file(READ "${INPUT}" _module)
string(REGEX REPLACE "(^|\n)!llvm\\.module\\.flags[^\n]*" "\\1" _module "${_module}")
file(WRITE "${OUTPUT}" "${_module}")
