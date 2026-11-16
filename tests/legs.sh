# The legs of tests/run.sh (sourced by it; its helpers in scope). LARES-932, spec §5.

ALL_TAGS="T A M F C R E"
SMALL="empty one all256 digits"
MID="run65535 run65536 text packed"

# ---- check: sizes and `sigil check`, warnings read (W0257 exits 0) ----
if on check; then
    if python3 -I "$here/sizes.py" "$tome" >"$work/sizes" 2>&1; then pass "check sizes: $(tail -1 "$work/sizes")"
    else fail "check sizes: $(grep -m1 OVER "$work/sizes")"; fi
    checked() {   # checked <label> <dir> <file>
        ( cd "$2" && "$sigil_bin" check "$3" ) >"$work/check" 2>&1; local rc=$?
        if [ "$rc" -ne 0 ]; then fail "check $1: sigil check exited $rc: $(grep -m1 -iE 'error' "$work/check" | cut -c1-200)"
        elif grep -qiE 'warn|W[0-9]{4}' "$work/check"; then fail "check $1: a warning: $(grep -m1 -iE 'warn|W[0-9]{4}' "$work/check" | cut -c1-200)"
        else pass "check $1: $(grep -m1 '✓' "$work/check")"; fi
    }
    checked tome "$tome" src/lib.sg
    stage "$work/check-consumer" "$ALL_TAGS" "" all ""; checked consumer "$work/check-consumer" src/main.sg
fi

# ---- run: the interpreter; slow rows only with HAAGENTI_FULL=1 (spec §7 K3) ----
if on run; then
    program run rows 3600 "T A M F E" "" all "" "T A M F E"
    if [ "$full" -eq 1 ]; then
        program run corpus 36000 C "" all "" "C R"
        program run mib1 36000 V "" all mib1 "V W"
        program run mib8 72000 V "" all mib8 "V W"
    else
        program run small 3600 C "$SMALL" all "" "C R"
        program run mid 7200 C "$MID" all "" "C R"
        program run random 3600 C random none "" "C R"
        n="$(grep -c '^C ' "$exp/c.txt")"; k="$(want C "" all "" | awk '$2 != "random" || $4 == "none"' | wc -l)"
        blind "run: $((n - k)) of $n C rows (random above level none) and the 16 V/W rows not run: the interpreter's speed (spec §7 K3); HAAGENTI_FULL=1 runs them"
    fi
    [ "${HAAGENTI_TIMING:-0}" = 1 ] && timing run
fi

# ---- aot: every row ----
if on aot; then
    if [ ! -f "$home/runtime/libsigil_runtime.a" ]; then blind "aot: no runtime/libsigil_runtime.a under SIGIL_HOME=$home"
    else
        program aot rows 1800 "T A M F C E" "" all "" "$ALL_TAGS"
        program aot mib1 1800 V "" all mib1 "V W"
        program aot mib8 1800 V "" all mib8 "V W"
        [ "${HAAGENTI_TIMING:-0}" = 1 ] && timing aot
    fi
fi

# ---- wasm: every row; mib8 one level a program (README: node's heap) ----
wasm_runtime() {
    if ! command -v node >/dev/null; then blind "wasm: node is missing"; return 1; fi
    if [ -n "${SIGIL_WASM_RUNTIME:-}" ]; then rt="$SIGIL_WASM_RUNTIME"; [ -f "$rt" ] && return 0; blind "wasm: SIGIL_WASM_RUNTIME '$rt' is not a file"; return 1; fi
    local pin; pin="$(tr -d '[:space:]' <"$here/wasm-runtime.pin")"
    mkdir -p "$work/q/x"
    if { git -C "$work/q" init -q && timeout 300 git -C "$work/q" fetch -q --depth 1 "${QLIPHOTH_URL:-https://github.com/Daemoniorum-LLC/qliphoth.git}" "$pin" &&
         git -C "$work/q" archive FETCH_HEAD runtime | tar -x -C "$work/q/x"; } >"$work/q.log" 2>&1 && [ -f "$work/q/x/runtime/sigil_runtime.js" ]; then
        rt="$work/q/x/runtime/sigil_runtime.js"; return 0
    fi
    blind "wasm: the fetch of qliphoth $pin failed: $(tail -2 "$work/q.log" | tr '\n' ' ' | cut -c1-200)"; return 1
}

# The negative controls: the runner passes a status through; a real program's checksums come back
# exactly; a wrong expectation is caught; a read past the end traps (exit 3), never reads as a pass.
wasm_controls() {
    if node "$here/run-wasm-check.mjs" "$here/run-wasm.mjs" >"$work/rwc" 2>&1; then pass "wasm control: run-wasm-check.mjs, $(grep -c '^ok' "$work/rwc") cases"
    else fail "wasm control: run-wasm-check.mjs: $(grep -m1 WRONG "$work/rwc")"; fi
    local d="$work/wasm-z" rc
    stage "$d" Z "" all ""; exec_prog wasm "$d" "$d.out" 120; rc=$?
    if [ "$rc" -eq 0 ] && grep -qx 'Z cbf43926 024d0127' "$d.out"; then pass "wasm control: CRC-32 and Adler-32 through the tome on wasm: Z cbf43926 024d0127"
    else fail "wasm control: the checksum program gave rc=$rc: $(grep -m1 '^Z\|trap\|error' "$d.out")"; fi
    if grep -qx 'Z 00000000 024d0127' "$d.out"; then fail "wasm control: a wrong expectation matched"
    else pass "wasm control: the wrong expectation 'Z 00000000 024d0127' is not met"; fi
    stage "$d-w" ZW "" all ""; exec_prog wasm "$d-w" "$d-w.out" 120; rc=$?
    if [ "$rc" -eq 3 ] && grep -q '^Z cbf43926' "$d-w.out" && ! grep -q 'past the end' "$d-w.out"; then pass "wasm control: a read past the end traps (exit 3)"
    else fail "wasm control: a read past the end gave rc=$rc: $(tail -1 "$d-w.out" | cut -c1-200)"; fi
}

if on wasm && wasm_runtime; then
    echo "== wasm runtime: qliphoth $(tr -d '[:space:]' <"$here/wasm-runtime.pin"); node $(node --version)"
    wasm_controls
    program wasm rows 1800 "T A M F C E" "" all "" "$ALL_TAGS"
    program wasm mib1 1800 V "" all mib1 "V W"
    for lv in none fast default; do program wasm "mib8-$lv" 1800 V "" "$lv" mib8 "V W"; done
    [ "${HAAGENTI_TIMING:-0}" = 1 ] && timing wasm
fi
