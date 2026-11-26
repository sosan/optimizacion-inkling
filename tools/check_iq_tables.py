#!/usr/bin/env python3
"""Verify src/ik2/ik2_iq2_tables.py against third_party/ggml/ggml-common.h.

The ik2 engine carries transcribed copies of ggml's IQ codebooks (so its kernels do not
include ggml-common.h). This script is the guard that the copies have not drifted:

    python3 tools/check_iq_tables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML = ROOT / "third_party/ggml/ggml-common.h"
TABLES = ROOT / "src/ik2/ik2_iq2_tables.h"

# (macro in ik2, table in ggml-common.h, use hex literals?)
PAIRS = [
    ("IK2_IQ2XXS_GRID", "iq2xxs_grid", True),
    ("IK2_IQ2XS_GRID", "iq2xs_grid", True),
    ("IK2_IQ2S_GRID", "iq2s_grid", True),
    ("IK2_IQ1S_GRID_GPU", "iq1s_grid_gpu", True),
    ("IK2_KSIGNS_IQ2XS", "ksigns_iq2xs", False),
    ("IK2_KMASK_IQ2XS", "kmask_iq2xs", False),
]


def ggml_table(name: str, hexnums: bool) -> list[str]:
    s = GGML.read_text(encoding="utf-8")
    m = re.search(r"GGML_TABLE_BEGIN\(\w+,\s*" + re.escape(name) + r",\s*[^\)]+\)\s*\n(.*?)\nGGML_TABLE_END",
                  s, re.S)
    if not m:
        raise ValueError(f"table {name} not found in {GGML}")
    return re.findall(r"0x[0-9a-fA-F]+" if hexnums else r"\d+", m.group(1))


def ik2_table(macro: str) -> list[str]:
    s = TABLES.read_text(encoding="utf-8")
    m = re.search(r"#define " + re.escape(macro) + r"\s*\\\n((?:.*\\\n)*.*)", s)
    if not m:
        raise ValueError(f"macro {macro} not found in {TABLES}")
    body = m.group(1).replace("\\\n", " ")
    return re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body)


def main() -> int:
    ok = True
    for macro, name, hexnums in PAIRS:
        a, b = ik2_table(macro), ggml_table(name, hexnums)
        same = a == b
        print(f"{macro:20} ik2={len(a):5} ggml={len(b):5} {'OK' if same else 'MISMATCH'}")
        if not same:
            ok = False
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"  first difference at {i}: ik2={x} ggml={y}")
                    break
    if not re.search(r"#define IK2_IQ1S_DELTA 0\.125f", TABLES.read_text(encoding="utf-8")):
        print("IK2_IQ1S_DELTA missing or wrong")
        ok = False
    if not ok:
        return 1
    print("all IQ codebooks match ggml-common.h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
