#!/bin/bash
# =============================================================================
# Haagenti Comprehensive Benchmark Suite
# =============================================================================
#
# Usage:
#   ./scripts/benchmark-all.sh              # Run all benchmarks
#   ./scripts/benchmark-all.sh --quick      # Quick mode (fewer iterations)
#   ./scripts/benchmark-all.sh --compare    # Compare against baseline
#   ./scripts/benchmark-all.sh --stack X    # Run specific stack (compression, tensor, gpu, ml, distributed)
#   ./scripts/benchmark-all.sh --crate X    # Run specific crate benchmarks
#
# Environment:
#   HAAGENTI_BENCH_BASELINE=<dir>   # Directory containing baseline results
#   HAAGENTI_BENCH_OUTPUT=<dir>     # Output directory for results (default: target/criterion)
#   CARGO_INCREMENTAL=0             # Disable incremental (recommended)
#   RUSTFLAGS="-C target-cpu=native" # Enable native CPU features
#
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${HAAGENTI_BENCH_OUTPUT:-$PROJECT_ROOT/target/criterion}"
BASELINE_DIR="${HAAGENTI_BENCH_BASELINE:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PROJECT_ROOT/benches/results/$TIMESTAMP"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Benchmark groups
declare -A BENCH_GROUPS
BENCH_GROUPS=(
    ["compression"]="haagenti haagenti-zstd"
    ["tensor"]="haagenti-fragments haagenti-sparse haagenti-neural haagenti-adaptive"
    ["gpu"]="haagenti-cuda"
    ["ml"]="haagenti-learning haagenti-autoopt haagenti-importance haagenti-speculative haagenti-merging haagenti-latent-cache"
    ["distributed"]="haagenti-distributed haagenti-streaming haagenti-serverless haagenti-network"
)

# Parse arguments
QUICK_MODE=false
COMPARE_MODE=false
STACK=""
CRATE=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --compare)
            COMPARE_MODE=true
            shift
            ;;
        --stack)
            STACK="$2"
            shift 2
            ;;
        --crate)
            CRATE="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if $SHOW_HELP; then
    echo "Haagenti Comprehensive Benchmark Suite"
    echo ""
    echo "Usage:"
    echo "  $0              # Run all benchmarks"
    echo "  $0 --quick      # Quick mode (fewer iterations)"
    echo "  $0 --compare    # Compare against baseline"
    echo "  $0 --stack X    # Run specific stack"
    echo "  $0 --crate X    # Run specific crate"
    echo ""
    echo "Stacks: compression, tensor, gpu, ml, distributed"
    echo ""
    echo "Crates:"
    for stack in "${!BENCH_GROUPS[@]}"; do
        echo "  $stack: ${BENCH_GROUPS[$stack]}"
    done
    exit 0
fi

# =============================================================================
# Helper functions
# =============================================================================

print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}\n"
}

print_section() {
    echo -e "\n${BLUE}───────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────────────────────────${NC}\n"
}

run_bench() {
    local crate="$1"
    local extra_args="${2:-}"

    echo -e "${YELLOW}▶ Running benchmarks for: $crate${NC}"

    local bench_args="--bench"
    if $QUICK_MODE; then
        bench_args="$bench_args -- --quick"
    fi

    if $COMPARE_MODE && [[ -n "$BASELINE_DIR" ]]; then
        bench_args="$bench_args --baseline $BASELINE_DIR"
    fi

    cd "$PROJECT_ROOT"

    if cargo bench -p "$crate" $bench_args $extra_args 2>&1 | tee -a "$RESULTS_DIR/$crate.log"; then
        echo -e "${GREEN}✓ $crate benchmarks completed${NC}"
        return 0
    else
        echo -e "${RED}✗ $crate benchmarks failed${NC}"
        return 1
    fi
}

check_crate_has_benches() {
    local crate="$1"
    local crate_path="$PROJECT_ROOT/crates/$crate"

    if [[ -d "$crate_path/benches" ]] && [[ -n "$(ls -A "$crate_path/benches" 2>/dev/null)" ]]; then
        return 0
    fi
    return 1
}

# =============================================================================
# Main
# =============================================================================

print_header "Haagenti Comprehensive Benchmark Suite"

echo -e "${BLUE}Configuration:${NC}"
echo "  Project root:  $PROJECT_ROOT"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Results dir:   $RESULTS_DIR"
echo "  Quick mode:    $QUICK_MODE"
echo "  Compare mode:  $COMPARE_MODE"
if [[ -n "$BASELINE_DIR" ]]; then
    echo "  Baseline dir:  $BASELINE_DIR"
fi
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Set recommended environment
export CARGO_INCREMENTAL=0
export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"

echo -e "${BLUE}Environment:${NC}"
echo "  CARGO_INCREMENTAL=$CARGO_INCREMENTAL"
echo "  RUSTFLAGS=$RUSTFLAGS"
echo ""

# Determine which crates to benchmark
CRATES_TO_BENCH=()

if [[ -n "$CRATE" ]]; then
    # Single crate mode
    CRATES_TO_BENCH=("$CRATE")
elif [[ -n "$STACK" ]]; then
    # Stack mode
    if [[ -z "${BENCH_GROUPS[$STACK]:-}" ]]; then
        echo -e "${RED}Unknown stack: $STACK${NC}"
        echo "Available stacks: ${!BENCH_GROUPS[*]}"
        exit 1
    fi
    read -ra CRATES_TO_BENCH <<< "${BENCH_GROUPS[$STACK]}"
else
    # All stacks
    for stack in compression tensor gpu ml distributed; do
        read -ra stack_crates <<< "${BENCH_GROUPS[$stack]}"
        CRATES_TO_BENCH+=("${stack_crates[@]}")
    done
fi

# =============================================================================
# Run benchmarks
# =============================================================================

TOTAL_CRATES=${#CRATES_TO_BENCH[@]}
COMPLETED=0
FAILED=0
SKIPPED=0

print_section "Running Benchmarks"

for crate in "${CRATES_TO_BENCH[@]}"; do
    if check_crate_has_benches "$crate"; then
        if run_bench "$crate"; then
            ((COMPLETED++))
        else
            ((FAILED++))
        fi
    else
        echo -e "${YELLOW}⊘ Skipping $crate (no benchmarks found)${NC}"
        ((SKIPPED++))
    fi
done

# =============================================================================
# Summary
# =============================================================================

print_header "Benchmark Summary"

echo -e "${BLUE}Results:${NC}"
echo -e "  ${GREEN}Completed:${NC} $COMPLETED"
echo -e "  ${RED}Failed:${NC}    $FAILED"
echo -e "  ${YELLOW}Skipped:${NC}   $SKIPPED"
echo -e "  Total:     $TOTAL_CRATES"
echo ""

echo -e "${BLUE}Output:${NC}"
echo "  Criterion HTML: $OUTPUT_DIR"
echo "  Logs:          $RESULTS_DIR"
echo ""

# Create summary file
cat > "$RESULTS_DIR/summary.txt" << EOF
Haagenti Benchmark Summary
==========================
Timestamp: $(date)
Git commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
Git branch: $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

Configuration:
  Quick mode: $QUICK_MODE
  Compare mode: $COMPARE_MODE
  RUSTFLAGS: $RUSTFLAGS

Results:
  Completed: $COMPLETED
  Failed: $FAILED
  Skipped: $SKIPPED
  Total: $TOTAL_CRATES

Crates benchmarked:
$(printf '  - %s\n' "${CRATES_TO_BENCH[@]}")
EOF

echo -e "${GREEN}Summary written to: $RESULTS_DIR/summary.txt${NC}"

if [[ $FAILED -gt 0 ]]; then
    echo -e "\n${RED}Some benchmarks failed. Check logs for details.${NC}"
    exit 1
fi

echo -e "\n${GREEN}All benchmarks completed successfully!${NC}"
