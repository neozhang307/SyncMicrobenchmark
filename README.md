# Sync_Microbenchmark
# Abstract
This work aims at characterizing the synchronization methods in CUDA. It mainly includes two part:
1. Non-primizive Synchronization:
  * Implicit Barrier, i.e. overhead of launching a kernel, including the new kernel launch function introduced for Cooperative Groups.
  * Multi-GPU synchronization with OpenMP in single node.
2. Primitive Synchronization Methods in Nvidia GPUs introduced in CUDA 9.0:
  * Warp level, Thread block level and grid level synchronization.
  * Multi-grid synchronization for Multi-GPU synchronization.
# Installation
  TBC
# Citation
  This research will be published in IPDPS20 
