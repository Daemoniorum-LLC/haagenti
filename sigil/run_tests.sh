#!/bin/bash
# Haagenti Sigil Test Runner
# Runs all test cases using the Sigil interpreter

SIGIL_COMPILER="/home/crook/dev/sigil-lang/parser/target/release/sigil"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
TOTAL=0

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Haagenti Sigil Test Suite                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check compiler exists
if [ ! -f "$SIGIL_COMPILER" ]; then
    echo -e "${RED}❌ Error: Sigil compiler not found at $SIGIL_COMPILER${NC}"
    echo "Run: cd /home/crook/dev/sigil-lang/parser && cargo build --release"
    exit 1
fi

# Function to run a single test
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sg)
    local expected="${test_file%.sg}.expected"
    local temp_out=$(mktemp)
    local temp_err=$(mktemp)

    TOTAL=$((TOTAL + 1))

    # Run the test
    if "$SIGIL_COMPILER" run "$test_file" > "$temp_out" 2> "$temp_err"; then
        # Filter DEBUG lines from output
        grep -v "^DEBUG" "$temp_out" > "${temp_out}.filtered" 2>/dev/null || true
        mv "${temp_out}.filtered" "$temp_out"

        if [ -f "$expected" ]; then
            if diff -q "$temp_out" "$expected" > /dev/null 2>&1; then
                echo -e "  ${GREEN}✓ PASS${NC}: $test_name"
                PASS=$((PASS + 1))
            else
                echo -e "  ${RED}✗ FAIL${NC}: $test_name (output mismatch)"
                echo "    Expected:"
                sed 's/^/      /' "$expected"
                echo "    Got:"
                sed 's/^/      /' "$temp_out"
                FAIL=$((FAIL + 1))
            fi
        else
            echo -e "  ${YELLOW}⚠ WARN${NC}: $test_name (no .expected file)"
            echo "    Output was:"
            sed 's/^/      /' "$temp_out"
            PASS=$((PASS + 1))  # Count as pass if it ran without error
        fi
    else
        echo -e "  ${RED}✗ FAIL${NC}: $test_name (runtime error)"
        cat "$temp_err" "$temp_out" 2>/dev/null | grep -v "^DEBUG" | sed 's/^/      /'
        FAIL=$((FAIL + 1))
    fi

    rm -f "$temp_out" "$temp_err"
}

# Find and run all tests
echo -e "${BLUE}Running haagenti-core tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-core/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-hct tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-hct/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-lz4 tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-lz4/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-brotli tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-brotli/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-simd tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-simd/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-stream tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-stream/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-adaptive tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-adaptive/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-grpc tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-grpc/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-latent-cache tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-latent-cache/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-zstd tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-zstd/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-deflate tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-deflate/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-neural tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-neural/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

echo ""
echo -e "${BLUE}Running haagenti-sparse tests...${NC}"
for test_file in "$SCRIPT_DIR"/haagenti-sparse/tests/*.sg; do
    if [ -f "$test_file" ]; then
        run_test "$test_file"
    fi
done

# Summary
echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, $TOTAL total"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
