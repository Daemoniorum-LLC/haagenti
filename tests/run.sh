#!/usr/bin/env bash
# haagenti in Sigil, slice 1 (LARES-932, specs/sigil/SLICE-1-CODEC-DEFLATE.md §5) on every executor.
#
#   exit 0  PASS            every row looked at held
#   exit 1  FAIL            a row differed, a program failed, or a control did not hold
#   exit 2  COULD-NOT-LOOK  nothing failed, and some row or leg could not be looked at
#
# Legs (HAAGENTI_LEGS names a subset): check run aot wasm. No Rust runs: tests/expected/ and the
# generated data (tests/consumer/src/gen/) are tests/oracle's, committed.
#   check  file and rite sizes; `sigil check` of the tome and the consumer prints no error or warning
#   run    `sigil run`: every row but the slow ones (below); aot: every row, mib1 and mib8 included
#   wasm   `sigil wasm` under node and qliphoth's runtime at tests/wasm-runtime.pin, every row, and the
#          negative controls: tests/run-wasm-check.mjs, a must-equal (Z) and a must-trap program
# The consumer (tests/consumer/src) prints a line per row; each tag's lines must be exactly
# tests/expected/<tag>.txt's for the rows the program ran. It is staged outside the repo, beside a
# manifest naming the tome: the repo keeps one manifest (spec §1).
#
# `sigil run` interprets about 2×10^5 loop steps a second (spec §7 K3). Unless HAAGENTI_FULL=1 it does
# not run the C rows of `random` above level none, or any volume row: each is reported COULD-NOT-LOOK,
# with its count.
#
#   SIGIL=/path/to/sigil (default: sigil on PATH)    SIGIL_HOME: where runtime/libsigil_runtime.a is
#   SIGIL_WASM_RUNTIME: qliphoth's runtime/sigil_runtime.js; unset, fetched at the pin
#   HAAGENTI_TIMING=1: also time 1 MiB at Default, compress and decompress, per leg (information only)
set -u
here="$(cd "$(dirname "$0")" && pwd)"; tome="$(cd "$here/.." && pwd)"; exp="$here/expected"
failed=0; blind=0
pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; failed=1; }
blind() { echo "COULD-NOT-LOOK $*"; blind=1; }
finish() {
    echo "== END $(date -Is)"
    if [ "$failed" -eq 1 ]; then echo "RESULT: FAIL"; exit 1; fi
    if [ "$blind" -eq 1 ]; then echo "RESULT: COULD-NOT-LOOK"; exit 2; fi
    echo "RESULT: PASS"; exit 0
}
sigil_bin="$(command -v "${SIGIL:-sigil}" 2>/dev/null)"
[ -n "$sigil_bin" ] && [ -x "$sigil_bin" ] || { echo "COULD-NOT-LOOK no sigil compiler at '${SIGIL:-sigil}'"; echo "RESULT: COULD-NOT-LOOK"; exit 2; }
echo "== START $(date -Is)"; "$sigil_bin" --version 2>&1 | sed -n '1,3p'; sha256sum "$sigil_bin"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
export NO_COLOR=1
legs=" ${HAAGENTI_LEGS:-check run aot wasm} "
on() { case "$legs" in *" $1 "*) return 0 ;; esac; return 1; }
full=0; [ "${HAAGENTI_FULL:-0}" = 1 ] && full=1
home="${SIGIL_HOME:-$(cd "$(dirname "$sigil_bin")/../.." 2>/dev/null && pwd)}"

# stage <dir> <sections> <corpus> <levels> <volume>: the consumer beside a manifest, mode set.
stage() {
    rm -rf "$1"; mkdir -p "$1"; cp -r "$here/consumer/src" "$1/src"
    printf '[project]\nname = "haagenti-consumer"\nversion = "0.0.0"\nedition = "2025"\n\n[dependencies]\nhaagenti = { path = "%s" }\n' "$tome" >"$1/Sigil.toml"
    sed -i "s/MODE_SECTIONS: &str = \".*\"/MODE_SECTIONS: \&str = \"$2\"/; s/MODE_CORPUS: &str = \".*\"/MODE_CORPUS: \&str = \"$3\"/" "$1/src/mode.sg"
    sed -i "s/MODE_LEVELS: &str = \".*\"/MODE_LEVELS: \&str = \"$4\"/; s/MODE_VOLUME: &str = \".*\"/MODE_VOLUME: \&str = \"$5\"/" "$1/src/mode.sg"
}

# exec_prog <leg> <dir> <out> <seconds>: run the staged program; its exit, 200 if it did not build.
# The peak resident size of the run (KB) is left in $peak.
exec_prog() {
    local leg="$1" d="$2" out="$3" s="$4" cmd
    case "$leg" in
        run) cmd=("$sigil_bin" run --no-deprecation-warnings src/main.sg) ;;
        aot) ( cd "$home" && timeout 900 "$sigil_bin" compile "$d/src/main.sg" -o "$d.bin" ) >"$d.build" 2>&1 && [ -x "$d.bin" ] ||
                 { grep -m1 -iE 'error|could not' "$d.build" | cut -c1-200 >"$out"; return 200; }
             cmd=("$d.bin") ;;
        wasm) ( cd "$d" && timeout 900 "$sigil_bin" wasm --no-deprecation-warnings src/main.sg -o "$d.wasm" ) >"$d.build" 2>&1 && [ -s "$d.wasm" ] ||
                  { grep -m1 -iE 'error|could not|unresolved' "$d.build" | cut -c1-200 >"$out"; return 200; }
              cmd=(node "$here/run-wasm.mjs" "$rt" "$d.wasm") ;;
    esac
    ( cd "$d" && timeout "$s" python3 -I "$here/peak.py" "$out" "${cmd[@]}" ); local rc=$?
    peak="$(sed -n 's/^PEAK //p' "$out.peak" 2>/dev/null)"
    return $rc
}

# want <tag> <corpus> <levels> <volume>: the expected lines of the rows a program with this mode runs.
want() {
    grep -v '^#' "$exp/$(echo "$1" | tr 'TAMFCREVW' 'tamfcrevv').txt" | awk -v t="$1" -v c=" $2 " -v l=" $3 " -v v="$4" '
        $1 != t { next }
        t == "C" || t == "R" { if ((c == "  " || index(c, " " $2 " ")) && (l == " all " || index(l, " " $4 " "))) print; next }
        t == "F" { if (c == "  " || index(c, " " $2 " ")) print; next }
        t == "V" || t == "W" { if ($2 == v && (l == " all " || index(l, " " $4 " "))) print; next }
        { print }'
}

# judge <leg> <label> <out> <rc> <tags> <corpus> <levels> <volume>
judge() {
    local leg="$1" label="$2" out="$3" rc="$4" tags="$5" t n
    if [ "$rc" -eq 200 ]; then blind "$leg $label: did not build: $(head -1 "$out")"; return; fi
    if ! grep -qx 'ORACLE-BEGIN' "$out"; then fail "$leg $label: never started (rc=$rc): $(grep -v '^ *$' "$out" | tail -1 | cut -c1-200)"; return; fi
    for t in $tags; do
        want "$t" "$6" "$7" "$8" >"$work/want"; grep -a "^$t " "$out" >"$work/got"; n="$(wc -l <"$work/want")"
        if cmp -s "$work/want" "$work/got"; then pass "$leg $label $t: $n rows byte-identical to tests/expected"
        else fail "$leg $label $t: rows differ (want $n, got $(wc -l <"$work/got")):"; diff "$work/want" "$work/got" | grep -a '^[<>]' | head -4 | cut -c1-200; fi
    done
    if [ "$rc" -eq 0 ] && grep -qx 'ORACLE-END' "$out" && ! grep -qavE '^([TAMFCREVWXZ] .*|ORACLE-BEGIN|ORACLE-END|main returned \(\))$' "$out"; then
        pass "$leg $label: ended rc=0, no stray line${peak:+, peak resident ${peak} KB}"
    else fail "$leg $label: rc=$rc, or no ORACLE-END, or a stray line: $(grep -avE '^([TAMFCREVWXZ] .*|ORACLE-BEGIN|ORACLE-END|main returned \(\))$' "$out" | head -1 | cut -c1-200)"; fi
}

# program <leg> <label> <seconds> <sections> <corpus> <levels> <volume> <tags>
program() {
    local d="$work/$1-$2"; stage "$d" "$4" "$5" "$6" "$7"
    exec_prog "$1" "$d" "$d.out" "$3"; local rc=$?
    judge "$1" "$2" "$d.out" "$rc" "$8" "$5" "$6" "$7"
}

# timing <leg>: seconds to make mib1 (X), and to compress (XK) and decompress (XKD) it at Default.
timing() {
    local d="$work/$1-time" t0 a b c
    for s in X XK XKD; do
        stage "$d$s" "$s" "" all mib1; t0="$(date +%s.%N)"; exec_prog "$1" "$d$s" "$d$s.out" 7200 >/dev/null 2>&1
        eval "t_$s=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')"
    done
    echo "TIME $1 mib1 Default: compress $(awk -v a="$t_XK" -v b="$t_X" 'BEGIN { printf "%.2f", a - b }')s, decompress $(awk -v a="$t_XKD" -v b="$t_XK" 'BEGIN { printf "%.2f", a - b }')s (make the input: ${t_X}s; build time included in each)"
}

source "$here/legs.sh"
finish
