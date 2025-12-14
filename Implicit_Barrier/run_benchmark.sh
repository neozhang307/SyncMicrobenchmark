#!/bin/bash
# Implicit Barrier Benchmark Script
# Measures kernel launch overhead on NVIDIA GPUs

echo "============================================================"
echo "  Implicit Barrier Benchmark - Kernel Launch Overhead"
echo "============================================================"
echo ""

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME (Compute Capability: $GPU_CC)"
echo ""

# Run benchmark and capture output
OUTPUT=$(./ImplicitBarrier 2>&1)

echo "============================================================"
echo "  1. Empty Kernel Launch Overhead"
echo "============================================================"
echo ""
echo "Measures the pure overhead of launching an empty kernel."
echo "(Launches 128 empty kernels and measures average overhead)"
echo ""

# Extract Empty Kernel section only (between first and second separator line)
EMPTY_KERNEL=$(echo "$OUTPUT" | sed -n '/Empty Kernel/,/Fuse Sleep/p' | head -n -2)

echo "$EMPTY_KERNEL" | awk '
/^traditional_launch.*\t1\t48\t/ {
    printf "Traditional Launch (1 kernel):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  API Call Overhead:  %8.2f ns\n", $10
}
/^traditional_launch.*\t128\t48\t/ {
    printf "Traditional Launch (128 kernels):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  Per-kernel overhead: %7.2f ns  ★\n", $16
    printf "\n"
}
/^cooperative_launch.*\t1\t48\t/ {
    printf "Cooperative Launch (1 kernel):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  API Call Overhead:  %8.2f ns\n", $10
}
/^cooperative_launch.*\t128\t48\t/ {
    printf "Cooperative Launch (128 kernels):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  Per-kernel overhead: %7.2f ns  ★\n", $16
    printf "\n"
}
/^cuda_graph_launch.*\t1\t48\t/ {
    printf "CUDA Graph Launch (1 node):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  API Call Overhead:  %8.2f ns\n", $10
}
/^cuda_graph_launch.*\t128\t48\t/ {
    printf "CUDA Graph Launch (128 nodes):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  Per-kernel overhead: %7.2f ns  ★\n", $16
    printf "\n"
}
/^graph_replay.*\t1\t48\t/ {
    printf "Graph Replay (1 launch):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  API Call Overhead:  %8.2f ns\n", $10
}
/^graph_replay.*\t128\t48\t/ {
    printf "Graph Replay (128 launches):\n"
    printf "  Total Latency:      %8.2f ns\n", $12
    printf "  Per-launch overhead: %7.2f ns  ★\n", $16
    printf "\n"
}
'

echo "============================================================"
echo "  2. Sleep Kernel Test (Overhead with actual workload)"
echo "============================================================"
echo ""
echo "Tests overhead when kernel has actual work (5000 ns sleep)."
echo ""

# Extract Sleep Kernel section
SLEEP_KERNEL=$(echo "$OUTPUT" | sed -n '/Fuse Sleep/,/Test the influence/p' | head -n -2)

echo "$SLEEP_KERNEL" | awk '
/^traditional_launch.*1:16/ {
    printf "Traditional Launch:\n"
    printf "  Ideal workload:      %8.2f ns\n", $6
    printf "  Measured workload:   %8.2f ns\n", $7
    printf "  Launch overhead:     %8.2f ns  ★\n", $9
    printf "\n"
}
/^cooperative_launch.*1:16/ {
    printf "Cooperative Launch:\n"
    printf "  Ideal workload:      %8.2f ns\n", $6
    printf "  Measured workload:   %8.2f ns\n", $7
    printf "  Launch overhead:     %8.2f ns  ★\n", $9
    printf "\n"
}
/^cuda_graph_launch.*1:16/ {
    printf "CUDA Graph Launch:\n"
    printf "  Ideal workload:      %8.2f ns\n", $6
    printf "  Measured workload:   %8.2f ns\n", $7
    printf "  Launch overhead:     %8.2f ns  ★\n", $9
    printf "\n"
}
/^graph_replay.*1:16/ {
    printf "Graph Replay:\n"
    printf "  Ideal workload:      %8.2f ns\n", $6
    printf "  Measured workload:   %8.2f ns\n", $7
    printf "  Launch overhead:     %8.2f ns  ★\n", $9
    printf "\n"
}
'

echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo ""

# Extract and display summary
TRAD_OVERHEAD=$(echo "$EMPTY_KERNEL" | awk '/^traditional_launch.*\t128\t48\t/ {print $16}')
COOP_OVERHEAD=$(echo "$EMPTY_KERNEL" | awk '/^cooperative_launch.*\t128\t48\t/ {print $16}')
GRAPH_OVERHEAD=$(echo "$EMPTY_KERNEL" | awk '/^cuda_graph_launch.*\t128\t48\t/ {print $16}')
REPLAY_OVERHEAD=$(echo "$EMPTY_KERNEL" | awk '/^graph_replay.*\t128\t48\t/ {print $16}')
TRAD_SLEEP_OVH=$(echo "$SLEEP_KERNEL" | awk '/^traditional_launch.*1:16/ {print $9}')
COOP_SLEEP_OVH=$(echo "$SLEEP_KERNEL" | awk '/^cooperative_launch.*1:16/ {print $9}')
GRAPH_SLEEP_OVH=$(echo "$SLEEP_KERNEL" | awk '/^cuda_graph_launch.*1:16/ {print $9}')
REPLAY_SLEEP_OVH=$(echo "$SLEEP_KERNEL" | awk '/^graph_replay.*1:16/ {print $9}')

printf "%-25s %12s %12s %12s %12s\n" "" "Traditional" "Cooperative" "CUDA Graph" "Graph Replay"
printf "%-25s %12s %12s %12s %12s\n" "" "-----------" "-----------" "----------" "------------"
printf "%-25s %9.0f ns %9.0f ns %9.0f ns %9.0f ns\n" "Empty kernel overhead:" "$TRAD_OVERHEAD" "$COOP_OVERHEAD" "$GRAPH_OVERHEAD" "$REPLAY_OVERHEAD"
printf "%-25s %9.0f ns %9.0f ns %9.0f ns %9.0f ns\n" "With workload overhead:" "$TRAD_SLEEP_OVH" "$COOP_SLEEP_OVH" "$GRAPH_SLEEP_OVH" "$REPLAY_SLEEP_OVH"
echo ""
echo "★ = Key metrics (lower is better)"
echo ""
echo "Notes:"
echo "  - CUDA Graph: measures per-kernel overhead within a graph (1 vs 128 kernels)"
echo "  - Graph Replay: measures cudaGraphLaunch overhead (1 vs 128 launches of same graph)"
echo ""
