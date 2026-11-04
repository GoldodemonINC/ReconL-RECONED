#!/usr/bin/env bash
#
# reconl-bench resolution/shadow/backends sweep
#
# Tests 120+ FPS target across:
#   Resolutions: 12K, 6K, 4K, 2K, 1080p
#   Shadow configs: matched (shadow = frame res) and downscaled (shadow < frame res)
#   Backends: igpu, cpu, cpu+igpu, cpu+disk, cpu+igpu+disk
#
# Usage: ./run_resolution_sweep.sh [OUTPUT_DIR]
#
# Each run prints its fingerprint + steady-state results.
# Results are appended to results.tsv in the output directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="${SCRIPT_DIR}/../../target/release/reconl-bench.exe"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/sweep-results}"
RESULTS="${OUTPUT_DIR}/results.tsv"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------ helpers
# resolution name -> WxH
res() {
    case "$1" in
        12k)   echo "12288x6480" ;;
        6k)    echo "6144x3456"  ;;
        4k)    echo "3840x2160"  ;;
        2k)    echo "2560x1440"  ;;
        1080)  echo "1920x1080"  ;;
        480)   echo "960x540"    ;;
        *)     echo "unknown" ;;
    esac
}

# shadow resolution name -> texel budget in bytes (W*H*4)
shadow_budget() {
    local w h
    case "$1" in
        12k)   w=12288; h=6480 ;;
        6k)    w=6144;  h=3456  ;;
        4k)    w=3840;  h=2160  ;;
        2k)    w=2560;  h=1440  ;;
        1080)  w=1920;  h=1080  ;;
        720)   w=1280;  h=720   ;;
        480)   w=960;   h=540   ;;
        *)     echo "0" ;;
    esac
    echo $(( w * h * 4 ))
}

# frame budget in ms for a target FPS
target_ms() {
    # Use 0 to disable tier ladder interference for accurate baselines
    echo 0
}

# ------------------------------------------------------------------ build bench
if [ ! -f "$BENCH" ]; then
    echo "== building reconl-bench =="
    cd "$SCRIPT_DIR"/..
    cargo build --release -p reconl-bench 2>&1 | tail -5
    cd "$SCRIPT_DIR"
    BENCH="$(ls -t "$SCRIPT_DIR"/../../target/release/reconl-bench.exe 2>/dev/null | head -1)"
    BENCH="${SCRIPT_DIR}/../../target/release/$BENCH"
    if [ ! -f "$BENCH" ]; then
        echo "ERROR: reconl-bench not found — did cargo build succeed?"
        exit 1
    fi
fi

# ------------------------------------------------------------------ header
cat > "$RESULTS" <<'EOF'
# ReconL resolution/shadow/backends sweep
# target: 120+ FPS (--target-ms=0 for accurate baselines)
# date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
#
# columns: backend_desc  tier  resolution  frame_WxH  shadow_mode  shadow_cascades  shadow_budget  warmup  frames  fps  total_ms  shadow_ns  alloc_per_frame  result
#
EOF

echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$RESULTS"
echo "# target fps: 120 (--target-ms=0)" >> "$RESULTS"

# ------------------------------------------------------------------ run one config
# $1 = backend flag (--backend=...)
# $2 = tier flag (--tier=...)
# $3 = adapter flag (--adapter=... or empty)
# $4 = spill flag (--spill=1 or empty)
# $5 = spill-dir flag (--spill-dir=... or empty)
# $6 = desc label (e.g. "igpu", "cpu", "cpu+igpu")
# $7 = resolution name (12k, 6k, 4k, 2k, 1080)
# $8 = shadow name (12k, 6k, 4k, 2k, 1080, 480)
# $9 = shadow mode (on, cached)
run_config() {
    local backend_flag="$1"
    local tier_flag="$2"
    local adapter_flag="$3"
    local spill_flag="$4"
    local spill_dir_flag="$5"
    local label="$6"
    local res_name="$7"
    local shad_name="$8"
    local shad_mode="$9"

    local res_wh shad_budget frames warmup
    res_wh="$(res "$res_name")"
    shad_budget="$(shadow_budget "$shad_name")"
    frames=30
    warmup=5

    # 12K and 6K are expensive — fewer frames, more warmup
    case "$res_name" in
        12k) frames=15; warmup=10 ;;
        6k)  frames=20; warmup=8  ;;
        4k)  frames=30; warmup=5  ;;
        *)   frames=60; warmup=5  ;;
    esac

    # Use 3x shadow downsampling (shadow res = frame res / 3)
    # and 1 cascade for 1080p, 2 cascades for everything else.
    local shad_cascades=2
    case "$res_name" in
        1080) shad_cascades=1 ;;
    esac

    echo "------------------------------------------------------------------------"
    echo "== $label | $res_name frame ($res_wh) | $shad_name shadows ($shad_mode) =="

    local trace_file
    trace_file="$(mktemp -u "${OUTPUT_DIR}/trace-${label}-${res_name}-${shad_name}-${shad_mode}-XXXXXX.csv")"

    local rc=0
    local output
    output=$(
        "$BENCH" \
            $backend_flag \
            $tier_flag \
            ${adapter_flag:+ "$adapter_flag"} \
            ${spill_flag:+ "$spill_flag"} \
            ${spill_dir_flag:+ "$spill_dir_flag"} \
            --resolution="$res_wh" \
            --shadows="$shad_mode" \
            --shadow-cascades="$shad_cascades" \
            --shadow-budget="${shad_budget}" \
            --target-ms="$(target_ms)" \
            --frames="$frames" \
            --warmup="$warmup" \
            --trace="$trace_file" \
            2>&1 || rc=$?
    )

    echo "$output"

    # Parse key numbers from output
    local fps_str total_str shadow_str alloc_str final_tier result
    fps_str=$(echo "$output" | sed -n 's/.*(\([0-9.]*\) fps at the mean).*/\1/p')
    fps_str=${fps_str:-NA}
    total_str=$(echo "$output" | sed -n 's/.*avg \([0-9.]*\).*/\1/p')
    total_str=${total_str:-NA}
    shadow_str=$(echo "$output" | sed -n 's/.*shadow.*pass \(—[^ ]*\).*/\1/p')
    shadow_str=${shadow_str:-NA}
    alloc_str=$(echo "$output" | sed -n 's/.*(\([0-9.]*\) per frame).*/\1/p')
    alloc_str=${alloc_str:-NA}
    final_tier=$(echo "$output" | grep -oP 'tier \K[^ ]+' | tail -1)
    final_tier=${final_tier:-NA}

    # Determine pass/fail against 120 FPS
    if [ "$fps_str" = "NA" ] || [ "$fps_str" = "—" ]; then
        result="no-instrumentation"
    elif awk "BEGIN {exit !($fps_str >= 120)}" 2>/dev/null; then
        result="PASS-120"
    else
        result="FAIL-120"
    fi

    # Append to results (tab-separated)
    {
        echo -n "$label"
        echo -n $'\t'
        echo -n "$res_name"
        echo -n $'\t'
        echo -n "$res_wh"
        echo -n $'\t'
        echo -n "$shad_name"
        echo -n $'\t'
        echo -n "$shad_mode"
        echo -n $'\t'
        echo -n "$shad_cascades"
        echo -n $'\t'
        echo -n "$shad_budget"
        echo -n $'\t'
        echo -n "$warmup"
        echo -n $'\t'
        echo -n "$frames"
        echo -n $'\t'
        echo -n "${fps_str:-NA}"
        echo -n $'\t'
        echo -n "${total_str:-NA}"
        echo -n $'\t'
        echo -n "${shadow_str:-NA}"
        echo -n $'\t'
        echo -n "${alloc_str:-NA}"
        echo -n $'\t'
        echo -n "$result"
        echo -n $'\t'
        echo -n "$final_tier"
        echo
    } >> "$RESULTS"

    rm -f "$trace_file"

    if [ $rc -ne 0 ]; then
        echo "  !! reconl-bench exited with code $rc"
    fi
    echo
}

# ------------------------------------------------------------------ run all combos
echo "################################################################################"
echo "# ReconL resolution/shadow/backends sweep"
echo "# target: 120+ FPS (--target-ms=0)"
echo "# $(date)"
echo "################################################################################"
echo

# ----- Test 1: Variable shadow quality (shadow < frame res) -----
echo "################################################################################"
echo "# TEST 1: Variable shadow quality — does 3x shadow downscaling hit 120 FPS?"
echo "################################################################################"
echo    # 12K frame + 4K shadows (3x downscale)
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "12k" "4k" "on"

# 6K frame + 2K shadows (3x downscale)
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "6k" "2k" "on"

# 4K frame + 1080p shadows (3x downscale)
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "4k" "1080" "on"

# 2K frame + 720p shadows (3x downscale)
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "2k" "720" "on"

# 1080 frame + 480p shadows (3x downscale)
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "1080" "480" "on"

# ----- Test 2: Same resolution shadows (shadow = frame res) -----
echo "################################################################################"
echo "# TEST 2: Same-resolution shadows — can each res hit 120 FPS with matching shadows?
# (these are the extreme case; shadow budget = frame budget)"
echo "################################################################################"
echo    # Same-res shadows (shadow budget = frame budget — extreme case)
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "12k" "12k" "on"
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "6k" "6k" "on"
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "4k" "4k" "on"
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "2k" "2k" "on"
run_config "--backend=auto" "--tier=auto" "" "" "" "auto" "1080" "1080" "on"

# ----- Test 3: Backends at 1080p (practical test point) -----
echo "################################################################################"
echo "# TEST 3: Backends at 1080p — IGPU/CPU/CPU+IGPU/CPU+DISK/CPU+IGPU+DISK"
echo "# 1080p + 1080p shadows (same-res) and 1080p + 480p shadows (downscaled)"
echo "################################################################################"
echo

# IGPU: D3D11 on integrated
run_config "--backend=d3d11" "--tier=auto" "--adapter=integrated" "" "" "igpu" "1080" "480" "on"

# CPU: SoftCPU at T2
run_config "--backend=soft-cpu" "--tier=t2" "" "" "" "cpu" "1080" "480" "on"

# CPU + IGPU: auto tier, let ladder offload CPU -> IGPU when over target
run_config "--backend=auto" "--tier=auto" "--adapter=integrated" "" "" "cpu+igpu" "1080" "480" "on"

# CPU + DISK: SoftCPU at T4 with spill
run_config "--backend=soft-cpu" "--tier=t4" "--spill=1" "--spill-dir=${OUTPUT_DIR}/spill" "" "cpu+disk" "1080" "480" "cached"

# CPU + IGPU + DISK: auto tier, IGPU adapter, with spill on T4
run_config "--backend=auto" "--tier=auto" "--adapter=integrated" "--spill=1" "--spill-dir=${OUTPUT_DIR}/spill" "cpu+igpu+disk" "1080" "480" "cached"

# ----- Summary -----
echo "################################################################################"
echo "# Summary"
echo "################################################################################"
echo
echo "Results written to: $RESULTS"
echo
echo "== Quick summary (120 FPS gate) =="
grep -E 'PASS-120|FAIL-120|no-instrumentation' "$RESULTS" | \
    awk -F'\t' '{printf "  %-20s %-8s %-10s %-12s %s\n", $1, $2, $3, $7, $13}'

echo
echo "Full results:"
cat "$RESULTS"
