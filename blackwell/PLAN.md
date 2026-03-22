# GEMM Optimization Plan

Current: dense_gemm_4 (Pair-UMMA + Cluster) → Target: dense_gemm.py (~1700 TFLOPS)

## Versions Overview

| Version | File | Description | Status |
|---------|------|-------------|--------|
| dense_gemm | `dense_gemm.py` | Full-featured production kernel | ~1700 TFLOPS |
| dense_gemm_1 | `dense_gemm_1.py` | Low-level mbarrier API | ~340 TFLOPS |
| dense_gemm_2 | `dense_gemm_2.py` | Pipeline API (4-stage) | ~620 TFLOPS |
| dense_gemm_3 | `dense_gemm_3.py` | + Cluster support | ~620 TFLOPS |
| dense_gemm_4 | `dense_gemm_4.py` | + Pair-UMMA (CtaGroup.TWO) | ~620 TFLOPS |

## Benchmark Results (8192x8192x4096)

| Kernel | TFLOPS | Expected |
|--------|--------|----------|
| torch.matmul (cuBLAS) | ~1700 | - |
| dense_gemm.py | ~1730 | - |
| dense_gemm_4.py | 634.07 | ~1100 |
| dense_gemm_3.py | ~620 | ~750 |
| dense_gemm_2.py | ~620 | ~620 |
| dense_gemm_1.py | ~340 | ~340 |

## Analysis: Why aren't we seeing gains?

### dense_gemm_3 (Cluster Support)
- Added `cluster_shape_mn = (2, 1)` 
- But `is_a_mcast=False, is_b_mcast=False` (no multicast effective)
- Cluster only helps when A/B need to be shared across CTAs

### dense_gemm_4 (Pair-UMMA)
- Added `CtaGroup.TWO` for 2-CTA MMA
- `use_2cta_instrs = True` (verified)
- But TFLOPS unchanged - something else is bottleneck

### Possible Bottlenecks
1. **SMEM bandwidth**: Writing to SMEM is still the bottleneck
2. **MMA efficiency**: Not fully utilizing MMA units
3. **Register pressure**: Too many registers per thread
4. **ILP**: Not enough instruction-level parallelism

## Next Steps

### Option 1: Debug Performance Bottleneck
- Add profiling to see where time is spent
- Compare kernel launches with dense_gemm.py
- Check occupancy, register usage, SMEM conflicts

### Option 2: Try TMA Store
- dense_gemm.py uses TMA Store, we use SIMT autovec store
- TMA Store bypasses SMEM for C output
- **Challenge**: TMA Store requires proper tensor shapes with Rest dimensions
- Error: `local_tile` with proj=(None,None,None) fails on 2D input

### Option 3: Larger Cluster Shape
- Try cluster_shape_mn = (2, 2) for 4x parallelism
- Need to verify cluster configuration is valid

## Key Code Comparisons

### dense_gemm.py TMA Store Setup
```python
# gC_mnl has shape (bM, bN, RestM, RestN, RestL)
gC_mnl = cute.local_tile(
    mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
)
tCgC = thr_mma.partition_C(gC_mnl)  # Shape: (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
```

### Our Attempt (Failed)
```python
# This fails - local_tile can't create 5D from 2D with proj=(None,None,None)
gC_tma = cute.local_tile(mC_mnl, (256, 256, 0), (None, None, None))
```

## Working Versions

| File | TFLOPS | Notes |
|------|--------|-------|
| dense_gemm_4.py | 634 | Pair-UMMA + Cluster + SIMT Store |
| dense_gemm_3.py | ~620 | Cluster + SIMT Store |
| dense_gemm_2.py | ~620 | 4-stage pipeline |
| dense_gemm_1.py | ~340 | Low-level mbarrier |
