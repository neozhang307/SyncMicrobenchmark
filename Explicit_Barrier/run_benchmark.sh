#!/bin/bash
# Explicit Barrier Benchmark Script
# Measures explicit synchronization overhead on NVIDIA GPUs

echo "============================================================"
echo "  Explicit Barrier Benchmark - Synchronization Overhead"
echo "============================================================"
echo ""

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
SM_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME (Compute Capability: $GPU_CC)"
echo ""

# Run IntraSM benchmark
echo "============================================================"
echo "  1. Intra-SM Synchronization Latency"
echo "============================================================"
echo ""
echo "Measures synchronization latency within a single SM."
echo "Tests block sync and tile shuffle operations."
echo ""

INTRA_OUTPUT=$(./BenchmarkIntraSM 2>&1)

# Extract block sync latency (single SM)
echo "Block Synchronization Latency (cycles per sync):"
echo "$INTRA_OUTPUT" | awk '
BEGIN { started=0 }
/^method.*GPUCount.*rep.*blk.*thrd.*m\(avgcycle\)/ {
    started=1; next
}
started==1 && /^blocksync/ {
    printf "  %4d threads: %8.2f cycles\n", $5, $6
}
/^tile_shufl/ { started=0 }
'

echo ""
echo "Throughput Test (sync operations per cycle):"
echo "$INTRA_OUTPUT" | awk '
BEGIN { OFS="\t" }
/^method.*GPUCount.*rep.*blk.*thrd.*tile.*m\(ttl_latency\).*m\(thrput\)/ {
    next
}
/^tile_sync.*\t48\t/ {
    printf "  tile_sync   blocks=%4d threads=%4d: throughput=%6.3f ops/cycle\n", $4, $5, $8
}
/^blocksync.*\t48\t/ {
    printf "  blocksync   blocks=%4d threads=%4d: throughput=%6.3f ops/cycle\n", $4, $5, $8
}
' | head -20

echo ""
echo "============================================================"
echo "  2. Inter-SM Synchronization (Grid Sync)"
echo "============================================================"
echo ""
echo "Measures grid synchronization latency across all SMs."
echo "Uses cooperative launch with grid_group.sync()."
echo ""

INTER_OUTPUT=$(./BenchmarkInterSM 2>&1)

echo "Grid Sync Latency (nanoseconds per sync):"
echo "$INTER_OUTPUT" | awk '
/^grid_sync/ {
    blocks = $5
    threads = $6
    basic_lat = $7
    more_lat = $9
    avg_instr = $11
    printf "  blocks=%4d threads=%4d: avg_sync_latency=%8.2f ns\n", blocks, threads, avg_instr
}
' | head -15

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo ""

# Extract key metrics (blk=1 for single-SM latency test)
BLOCK_SYNC_32=$(echo "$INTRA_OUTPUT" | awk '/^blocksync/ && $4==1 && $5==32 {print $6; exit}')
BLOCK_SYNC_1024=$(echo "$INTRA_OUTPUT" | awk '/^blocksync/ && $4==1 && $5==1024 {print $6; exit}')
GRID_SYNC_MIN=$(echo "$INTER_OUTPUT" | awk '/^grid_sync/ {print $11}' | sort -n | head -1)
GRID_SYNC_MAX=$(echo "$INTER_OUTPUT" | awk '/^grid_sync/ {print $11}' | sort -n | tail -1)

printf "%-35s %15s\n" "Metric" "Value"
printf "%-35s %15s\n" "-----------------------------------" "---------------"
printf "%-35s %12.2f cycles\n" "Block sync (32 threads):" "$BLOCK_SYNC_32"
printf "%-35s %12.2f cycles\n" "Block sync (1024 threads):" "$BLOCK_SYNC_1024"
printf "%-35s %12.2f ns\n" "Grid sync (min):" "$GRID_SYNC_MIN"
printf "%-35s %12.2f ns\n" "Grid sync (max):" "$GRID_SYNC_MAX"
echo ""
echo "Note: Block sync is measured in GPU cycles"
echo "      Grid sync is measured in nanoseconds"
echo ""
