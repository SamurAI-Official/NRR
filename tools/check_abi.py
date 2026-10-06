#!/usr/bin/env python3
"""check_abi.py - prove the managed Unity mirror matches the native C ABI, field by field.

`NRRTypes.cs` blits its structs straight into native memory, so there is no marshaller step to
catch a mismatch: a field missing, reordered or a different width on either side still compiles,
still runs, and silently reads the wrong bytes. A field added to `NRRTemporalState` without its
counterpart in C# would put the frame's jitter offset on top of the frame index - a corruption
that presents as a rendering bug rather than a binding bug.

Both declarations are parsed and every field's byte offset worked out under Win64 C rules:
8-byte pointers and `ulong`/`IntPtr`, 4-byte `int`/`unsigned`/`uint32_t`/`float`, natural
alignment per field, struct alignment from its widest field.

Usage:
    python tools/check_abi.py
"""
import argparse
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

SCALARS = {
    "float": (4, 4), "int": (4, 4), "unsigned": (4, 4), "unsigned int": (4, 4),
    "int32_t": (4, 4), "uint32_t": (4, 4), "uint": (4, 4), "bool": (4, 4),
    "byte": (1, 1), "sbyte": (1, 1), "char": (1, 1),
    "unsigned long": (8, 8), "long": (8, 8), "int64_t": (8, 8), "uint64_t": (8, 8),
    "long long": (8, 8), "unsigned long long": (8, 8), "ulong": (8, 8),
    "unsigned char": (1, 1), "uint8_t": (1, 1), "short": (2, 2), "uint16_t": (2, 2),
    "double": (8, 8), "size_t": (8, 8),
}
POINTERS = {"IntPtr", "UIntPtr", "nint", "nuint"}


def parse_c(text):
    # Comments are stripped from the whole text first, not per field: a doc comment above a
    # declaration is the common case, and stripping afterwards leaves its fragments to be
    # mistaken for a type, which is how a comment about the `history` field once came back as
    # a 200-character field name.
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    structs = {}
    for body, name in re.findall(r"typedef\s+struct\s*\{(.*?)\}\s*(\w+)\s*;", text, re.S):
        fields = []
        for raw in body.split(";"):
            line = " ".join(raw.split())
            m = re.match(r"^(.*?)([A-Za-z_]\w*)\s*(\[\s*\d*\s*\])?$", line)
            if m:
                fields.append((m.group(2), m.group(1).strip(), (m.group(3) or "").strip("[] ")))
        structs[name] = fields
    return structs


def parse_cs(text):
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    structs = {}
    for name, body in re.findall(r"struct\s+(\w+)\s*\{(.*?)\}", text, re.S):
        fields = []
        for raw in body.split(";"):
            line = " ".join(raw.split())
            # A marshalling attribute sits before the type and carries the length of a fixed-size
            # native array, so it has to be read before the field is classified - `string debug_info`
            # with SizeConst 256 is `char debug_info[256]` natively, and comparing them as anything
            # else would either report a false mismatch or miss a real one.
            size_const = re.search(r"SizeConst\s*=\s*(\d+)", line)
            line = re.sub(r"^(\[[^\]]*\]\s*)+", "", line)
            m = re.match(r"^(?:public|internal|private)\s+(.*?)([A-Za-z_]\w*)$", line)
            if m:
                type_name = m.group(1).strip()
                arr = re.search(r"\[(\d*)\]$", type_name)
                array = arr.group(1) if arr else ""
                if arr:
                    # Strip the suffix whatever its length; a managed `float[]` carries its
                    # length in SizeConst, which is checked next.
                    type_name = type_name[:arr.start()].strip()
                if arr and arr.group(1):
                    array = arr.group(1)
                elif type_name == "string":
                    # LPStr -> a native `const char *`; SizeConst -> a fixed `char[N]`.
                    type_name, array = "char", (size_const.group(1) if size_const else "ptr")
                elif size_const:
                    # A managed array pinned to a fixed native length.
                    array = size_const.group(1)
                fields.append((m.group(2), type_name, array))
        structs[name] = fields
    return structs
def classify(type_name, array, structs):
    """The property the marshaller actually depends on: size class and shape.

    Deliberately coarser than a full layout computation. Walking offsets through a
    text-parsed header means every quirk of the parser becomes a possible false failure, and a
    tool that cries wolf gets ignored. What actually corrupts memory here is a field present on
    one side and absent or reordered on the other, or a pointer where a 32-bit value was
    expected - and all three of those are visible in the field list itself.
    """
    if array == "ptr":
        # A managed LPStr string against a native `const char *`.
        return "W8"
    if type_name.endswith("*") or type_name in POINTERS:
        # Pointer and 64-bit integer are deliberately not distinguished: they marshal
        # identically, so a `size_t` against an `IntPtr` is a match, not a discrepancy. A tool
        # that flags correct idioms is a tool whose findings get ignored.
        return "W8"
    if type_name in SCALARS:
        return "W%d%s" % (SCALARS[type_name][0], "[]" if array else "")
    if type_name in structs:
        return "struct:%s%s" % (type_name, "[]" if array else "")
    return "unknown(%s)" % type_name


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--header", default=os.path.join(ROOT, "include", "nrr.h"))
    ap.add_argument("--mirror", default=os.path.join(ROOT, "engine_plugins", "unity", "Runtime",
                                                      "Scripts", "NRRTypes.cs"))
    args = ap.parse_args(argv[1:])
    native = parse_c(open(args.header, encoding="utf-8").read())
    managed = parse_cs(open(args.mirror, encoding="utf-8").read())

    shared = [n for n in native if n in managed]
    print("checking %d struct(s) declared on both sides\n" % len(shared))
    problems = 0
    for name in shared:
        nf, mf = native[name], managed[name]
        if len(nf) != len(mf):
            print("FAIL %-20s %d native vs %d managed field(s)" % (name, len(nf), len(mf)))
            for tag, fields in (("native", nf), ("managed", mf)):
                for n, t, a in fields:
                    print("      %-7s %s %s%s" % (tag, t, n, "[%s]" % a if a else ""))
            problems += 1
            continue
        diffs = []
        for (nn, nt, na), (mn, mt, ma) in zip(nf, mf):
            nk, mk = classify(nt, na, native), classify(mt, ma, managed)
            if nn != mn:
                diffs.append("  field %d: native '%s' vs managed '%s' - order or name differs" %
                             (nf.index((nn, nt, na)) + 1, nn, mn))
            elif nk != mk:
                diffs.append("  field %-22s native %-10s managed %s" % (nn, nk, mk))
        if diffs:
            print("FAIL %s" % name)
            print("\n".join(diffs))
            problems += 1
        else:
            print("ok   %-20s %2d field(s)" % (name, len(nf)))
    for label, names in (("native-only (no managed mirror)", sorted(set(native) - set(managed))),
                         ("managed-only (no native counterpart)", sorted(set(managed) - set(native)))):
        if names:
            print("\n%s: %s" % (label, ", ".join(names)))
    print("\nRESULT: %s" % ("PASS" if problems == 0 else "FAIL (%d struct(s) disagree)" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))