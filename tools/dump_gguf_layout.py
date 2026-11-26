#!/usr/bin/env python3
"""Dump the GGUF *header* of a quant (metadata + tensor types/shapes) without
downloading the weight data. Uses HTTP Range requests so a 10-20 GB file costs
a few MB.

Works on the small/missing quants (UD-IQ2_XXS, UD-Q2_K_XL, UD-IQ3_XXS,
UD-Q8_K_XL, ...) from https://huggingface.co/unsloth/Inkling-2-Mini-GGUF.

Usage:
  python tools/dump_gguf_layout.py                       # all known quants, both layouts
  python tools/dump_gguf_layout.py UD-IQ2_XXS UD-Q2_K_XL # selected quants
  python tools/dump_gguf_layout.py --out report.json     # save full JSON (upload this)
  python tools/dump_gguf_layout.py --rev <sha>           # pin a revision

The tool prints a compact per-quant summary to stdout: architecture, total
tensors, the distinct ggml tensor types with counts, and which types the ik2
engine does not decode (compared against tools/ik2_supported_types.py if
present, else a built-in list). Upload the JSON with:
  gh gist create report.json   (or attach it to the upstream issue #19)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = "unsloth/Inkling-2-Mini-GGUF"
BRANCH = "main"
QUANTS = ["UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-IQ4_XS",
          "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_M", "UD-Q4_K_XL", "UD-Q8_K_XL"]
# Types the ik2 engine actually decodes (src/ik2/ik2_kernels.cu): F32/F16/BF16, Q8_0,
# Q4_K/Q5_K/Q6_K, IQ2_XXS/IQ2_XS/IQ3_XXS/IQ1_S/IQ2_S/IQ4_XS, MXFP4, NVFP4 and Q2_0
# (the last through its dedicated expert kernels). Issue #19 added Q4_K, IQ1_S, IQ2_XXS,
# IQ2_XS, IQ2_S and MXFP4. Types with header geometry only (Q4_0, Q2_K, IQ3_S, ...) are
# intentionally NOT listed: knowing a block's byte size is not decoding it.
SUPPORTED = {0, 1, 8, 12, 13, 14, 16, 17, 18, 19, 22, 23, 30, 39, 40, 42}

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I8", 30: "BF16",
    32: "Q4_0_4_4", 33: "Q4_0_4_8", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4", 41: "Q4_0_8_8", 42: "Q2_0",
}
GGUF_META = {0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2), 4: ("u32", 4),
             5: ("i32", 4), 6: ("f32", 4), 7: ("bool", 1), 8: ("string", None), 9: ("array", None),
             10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8)}


def url_for(filename: str, rev: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{rev}/{filename}"


def fetch_prefix(url: str, want: int, chunk: int = 1 << 20) -> bytes:
    """Fetch the first `want` bytes via a Range request, growing until the header fits."""
    got = bytearray()
    while len(got) < want:
        start, end = len(got), min(want, len(got) + chunk) - 1
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                part = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 416:  # file shorter than requested
                break
            raise
        if not part:
            break
        got += part
        if len(part) < end - start + 1:
            break
    return bytes(got)


class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0

    def take(self, n: int) -> bytes:
        if self.p + n > len(self.d):
            raise ValueError(f"header ends early at {self.p} (need {n} more bytes)")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self):
        t = self.u32()
        if t not in GGUF_META:
            raise ValueError(f"unknown metadata type {t} at {self.p - 4}")
        name, size = GGUF_META[t]
        if name == "string":
            return self.string()
        if name == "array":
            et = self.u32()
            count = self.u64()
            en, es = GGUF_META[et]
            if en == "string":
                return [self.string() for _ in range(count)]
            fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
                   "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}[en]
            return list(struct.unpack(f"<{count}{fmt}", self.take(es * count)))
        fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
               "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}[name]
        return struct.unpack(f"<{fmt}", self.take(size))[0]


def parse(data: bytes) -> dict:
    r = Reader(data)
    magic = struct.unpack("<I", r.take(4))[0]
    if magic != 0x46554747:
        raise ValueError(f"not GGUF (magic {magic:#x})")
    version, n_tensors, n_kv = struct.unpack("<IQQ", r.take(20))
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = r.value()
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        ndim = r.u32()
        shape = list(struct.unpack(f"<{ndim}Q", r.take(8 * ndim)))
        tid = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "shape": shape, "type_id": tid,
                        "type": GGML_TYPES.get(tid, f"type{tid}"), "offset": offset})
    return {"gguf_version": version, "n_tensors": n_tensors, "metadata": meta, "tensors": tensors,
            "header_bytes": r.p}


def compact(info: dict) -> dict:
    """Shrink metadata arrays (tokenizer vocab: 131k strings) to a count + sample so the report can be uploaded."""
    out = dict(info)
    meta = {}
    for k, v in info["metadata"].items():
        if isinstance(v, list) and len(v) > 32:
            meta[k] = {"array_len": len(v), "first_8": v[:8], "last_4": v[-4:]}
        else:
            meta[k] = v
    out["metadata"] = meta
    return out


def summarize(name: str, info: dict, had_to_grow: bool = False) -> dict:
    c = Counter(t["type"] for t in info["tensors"])
    unsupported = Counter(t["type"] for t in info["tensors"] if t["type_id"] not in SUPPORTED)
    groups = Counter()
    for t in info["tensors"]:
        n = t["name"]
        if ".ffn_" in n:
            groups["experts"] += 1
        elif "blk." in n:
            groups["dense/attn"] += 1
        else:
            groups["other"] += 1
    return {
        "file": name,
        "architecture": info["metadata"].get("general.architecture"),
        "gguf_version": info["gguf_version"],
        "n_tensors": info["n_tensors"],
        "groups": dict(groups),
        "tensor_types": dict(sorted(c.items(), key=lambda kv: -kv[1])),
        "unsupported_types": dict(sorted(unsupported.items(), key=lambda kv: -kv[1])),
        "inkling2_keys": sorted(k for k in info["metadata"] if k.startswith("inkling2.")),
        "header_bytes": info["header_bytes"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quants", nargs="*", help="quant names (default: all known)")
    ap.add_argument("--rev", default=BRANCH, help="HF revision/branch/commit")
    ap.add_argument("--out", type=Path, help="write the report (metadata + all tensors) as JSON, compact enough to upload")
    ap.add_argument("--full", action="store_true", help="keep large metadata arrays verbatim (tokenizer vocab: ~56 MB)")
    ap.add_argument("--both-layouts", action="store_true", help="also probe Old_Support/*.gguf_file")
    args = ap.parse_args()

    quants = args.quants or QUANTS
    report = {"repo": REPO, "revision": args.rev, "quants": {}}
    for q in quants:
        candidates = [f"Inkling-2-Mini-{q}.gguf"]
        if args.both_layouts or q in ("UD-Q4_K_XL", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-Q8_K_XL"):
            candidates.append(f"Old_Support/Inkling-2-Mini-{q}.gguf_file")
        for fn in candidates:
            url = url_for(fn, args.rev)
            print(f" probing {fn} ...", file=sys.stderr, flush=True)
            try:
                prefix = fetch_prefix(url, 16 << 20)
                info = parse(prefix)
            except Exception as e:
                print(f"  FAIL {fn}: {e}", file=sys.stderr)
                report["quants"][fn] = {"error": str(e)}
                continue
            s = summarize(fn, info)
            report["quants"][fn] = {**s, "full": info if args.full else compact(info)}
            print(f"\n=== {fn} ===")
            print(f"  arch={s['architecture']}  tensors={s['n_tensors']}  header={s['header_bytes']:,} B")
            print(f"  types: {s['tensor_types']}")
            if s["unsupported_types"]:
                print(f"  UNSUPPORTED by ik2: {s['unsupported_types']}")
            else:
                print("  all tensor types decodable today")
            print(f"  inkling2.* metadata keys: {len(s['inkling2_keys'])}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nfull report -> {args.out}  (upload this file to the issue)")

    bad = sum(1 for v in report["quants"].values() if v.get("unsupported_types"))
    print(f"\n{len(report['quants'])} files probed, {bad} with undecodable tensor types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
