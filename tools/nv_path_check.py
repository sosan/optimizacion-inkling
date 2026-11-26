"""tools/nv_path_check.py - an AMD-only change leaves the NVIDIA build's sources unchanged.

CONTRIBUTING asks AMD (HIP) work to keep NVIDIA's code path as it is, under #if defined(STRATA_USE_HIP).  This check
proves it as text, without a CUDA toolkit: every source file a branch changed is read the way the CUDA build reads it
(STRATA_USE_HIP undefined: the HIP-only branches and the STRATA_USE_HIP directives dropped, everything else kept) and
compared byte for byte with the base's file read the same way.  A file the branch added must be one the CUDA build
never reads: nothing outside its own STRATA_USE_HIP blocks (comments, blank lines, #pragma once), or a source that no
CUDA view #includes, or a test that CMakeLists.txt names only inside the AMD block (STRATA_HIP_IK2_SPIKE).

    python tools/nv_path_check.py [base]      # base: a commit (default origin/main); exit 1 on any difference

Recognised: #ifdef / #ifndef STRATA_USE_HIP, #if defined(STRATA_USE_HIP), #if !defined(STRATA_USE_HIP), their #else
and #endif.  An #elif on such a block is refused (write #else + #if).  Other conditionals pass through untouched.
"""
from __future__ import annotations

import re
import subprocess
import sys

SOURCE = re.compile(r"\.(cu|cuh|cpp|hpp|h|inc|c)$")
DIRECTIVE = re.compile(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)")


def kind(d: str, rest: str) -> str:
    """'hip' / 'cuda' when the condition is exactly STRATA_USE_HIP defined / undefined, else 'other'."""
    r = re.sub(r"//.*|/\*.*?\*/", "", rest).replace(" ", "").replace("\t", "")
    if d == "ifdef":
        return "hip" if r == "STRATA_USE_HIP" else "other"
    if d == "ifndef":
        return "cuda" if r == "STRATA_USE_HIP" else "other"
    if r in ("defined(STRATA_USE_HIP)", "STRATA_USE_HIP"):
        return "hip"
    if r in ("!defined(STRATA_USE_HIP)", "!STRATA_USE_HIP"):
        return "cuda"
    if "STRATA_USE_HIP" in r:
        raise ValueError(f"a STRATA_USE_HIP condition this check does not read: #{d}{rest}")
    return "other"


def nv_view(text: str, name: str = "") -> str:
    """The text the CUDA build sees as far as STRATA_USE_HIP decides (line structure otherwise kept)."""
    out, stack = [], []          # stack: [kind, in #else]
    for no, line in enumerate(text.splitlines(True), 1):
        m = DIRECTIVE.match(line)
        if m:
            d, rest = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef", "if"):
                stack.append([kind(d, rest), False])
                if stack[-1][0] != "other":
                    continue
            elif d == "elif":
                if not stack:
                    raise ValueError(f"{name}:{no}: #elif without #if")
                if stack[-1][0] != "other":
                    raise ValueError(f"{name}:{no}: #elif on a STRATA_USE_HIP block")
            elif d == "else":
                if not stack:
                    raise ValueError(f"{name}:{no}: #else without #if")
                stack[-1][1] = True
                if stack[-1][0] != "other":
                    continue
            elif d == "endif":
                if not stack:
                    raise ValueError(f"{name}:{no}: #endif without #if")
                k = stack.pop()[0]
                if k != "other":
                    continue
        # a line is dropped when any enclosing STRATA_USE_HIP block hides it from CUDA (directives of 'other' blocks
        # inside a hidden region are dropped with it)
        hidden = any((k == "hip" and not e) or (k == "cuda" and e) for k, e in stack)
        if m and m.group(1) in ("ifdef", "ifndef", "if") and not hidden:
            hidden = any((k == "hip" and not e) or (k == "cuda" and e) for k, e in stack[:-1])
        if not hidden:
            out.append(line)
    if stack:
        raise ValueError(f"{name}: {len(stack)} #if left open")
    return "".join(out)


def only_inert(text: str) -> bool:
    """True when the text holds nothing a compiler would compile: comments, blank lines, #pragma once."""
    t = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    return all(not l.strip() or l.strip() == "#pragma once" for l in t.splitlines())


def cmake_hip_only(name: str) -> bool:
    """True when CMakeLists.txt names the file, and only between `if(STRATA_HIP_IK2_SPIKE AND STRATA_ENABLE_HIP)` and
    the endif() at that if's indentation."""
    lines = open("CMakeLists.txt", encoding="utf-8").read().splitlines()
    inside, ind, seen = False, "", False
    for l in lines:
        if not inside and re.match(r"\s*if\(STRATA_HIP_IK2_SPIKE AND STRATA_ENABLE_HIP\)", l):
            inside, ind = True, l[: len(l) - len(l.lstrip())]
            continue
        if inside and l.startswith(ind + "endif()"):
            inside = False
            continue
        if name in l:
            if not inside:
                return False
            seen = True
    return seen


def git(*a: str) -> str:
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    bad = 0
    rows = git("diff", "--name-status", "--no-renames", base, "--").splitlines()
    rows += [f"A\t{p}" for p in git("ls-files", "--others", "--exclude-standard").splitlines()]
    # what the CUDA views #include (a new header / .inc that none of them includes is never compiled for CUDA)
    nv_includes = set()
    for row in rows:
        st, path = row.split("\t", 1)
        if st != "D" and SOURCE.search(path) and path.startswith(("src/", "include/", "tests/")):
            try:
                nv_includes |= set(re.findall(r'^\s*#\s*include\s*["<]([^">]+)[">]', nv_view(open(path).read()), re.M))
            except ValueError:
                pass
    for row in rows:
        status, path = row.split("\t", 1)
        if not SOURCE.search(path) or not (path.startswith(("src/", "include/", "tests/"))):
            continue
        try:
            head = nv_view(open(path, encoding="utf-8").read(), path) if status != "D" else None
            if status == "A":
                base_name = path.rsplit("/", 1)[-1]
                if only_inert(head):
                    ok, why = True, "HIP-only (its own guard)"
                elif path.startswith("tests/"):
                    ok = cmake_hip_only(path)
                    why = "a test only the AMD block of CMakeLists.txt builds" if ok else "a test the CUDA build may build"
                else:
                    ok = not any(i.endswith(base_name) for i in nv_includes)
                    why = "included only from STRATA_USE_HIP blocks" if ok else "a CUDA view includes it"
                print(f"{'ok  ' if ok else 'FAIL'} {path} (new): {why}")
            elif status == "D":
                ok = False
                print(f"FAIL {path}: deleted")
            else:
                ok = nv_view(git("show", f"{base}:{path}"), path) == head
                print(f"{'ok  ' if ok else 'FAIL'} {path}: {'CUDA view identical' if ok else 'the CUDA view differs'}")
        except ValueError as e:
            ok = False
            print(f"FAIL {e}")
        bad += not ok
    print(f"nv_path_check: {len(rows)} changed paths vs {base}, {bad} failing source files")
    return 1 if bad else 0


def _selftest() -> None:
    src = ("a\n#if defined(STRATA_USE_HIP)\nhip\n#if X\nhx\n#endif\n#else\nnv\n#endif  // STRATA_USE_HIP\n"
           "#ifndef STRATA_USE_HIP\nnv2\n#else\nhip2\n#endif\n#if !defined(STRATA_USE_HIP)\nnv3\n#endif\n#if Y\ny\n#endif\n")
    assert nv_view(src) == "a\nnv\nnv2\nnv3\n#if Y\ny\n#endif\n", nv_view(src)
    assert only_inert("// c\n#pragma once\n/* x\n y */\n\n")
    assert not only_inert("#pragma once\nint x;\n")
    for badsrc in ("#if defined(STRATA_USE_HIP)\n", "#ifdef STRATA_USE_HIP\n#elif Z\n#endif\n", "#if defined(STRATA_USE_HIP) && X\n#endif\n"):
        try:
            nv_view(badsrc)
            raise AssertionError(badsrc)
        except ValueError:
            pass


if __name__ == "__main__":
    _selftest()
    sys.exit(main())
