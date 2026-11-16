"""LARES-932's size ceilings (spec §1): 300 lines a file, 80 a rite, in src/ and tests/consumer/src/.

usage: python3 -I tests/sizes.py <repo>    exit 0 within the ceilings, 1 otherwise (each offender
printed). A rite runs from its `rite` line to the line where its braces balance. Generated data
scrolls (tests/consumer/src/gen/) are counted too.
"""
import pathlib
import sys

FILE, RITE = 300, 80


def rites(lines):
    """(name, first line, length) of each rite."""
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if (s.startswith("rite ") or s.startswith("☉ rite ")) and "{" in lines[i]:
            depth, j = 0, i
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0 and j >= i:
                    break
                j += 1
            yield s.split("(")[0], i + 1, j - i + 1
            i = j + 1
        else:
            i += 1


def main(root):
    bad = 0
    files = sorted(root.glob("src/**/*.sg")) + sorted(root.glob("tests/consumer/src/**/*.sg"))
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        if len(lines) > FILE:
            print(f"OVER {f.relative_to(root)}: {len(lines)} lines (ceiling {FILE})")
            bad = 1
        for name, at, n in rites(lines):
            if n > RITE:
                print(f"OVER {f.relative_to(root)}:{at} {name}: {n} lines (ceiling {RITE})")
                bad = 1
    print(f"{len(files)} files checked: {'within' if not bad else 'OVER'} the ceilings")
    return bad


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
