## Background
- `dense_gemm_1.py` is the first tutorial GEMM showing a minimal mbarrier-based implementation.
- `dense_gemm_2.py` advances the same workflow by adopting the Pipeline API (PipelineTmaUmma, PipelineUmmaAsync).
- Currently `dense_gemm_1.py` builds `gA`, `gB`, `gC` via `proj` so the rest axes are collapsed, and the variable names/comments drift between the two tutorials.

## Goals
1. Match `dense_gemm_1.py`’s local tiles to the shape/coordinate pattern used in `dense_gemm.py`—i.e., include the trailing `RestM`, `RestN`, `RestK` axes rather than projecting them away.
2. Propagate the shape change through every fragment/partition that observes `gA/gB/gC`, making sure the correct tile coordinates are used whenever we slice into the rest axes.
3. Harmonize the naming/column descriptions between `dense_gemm_1.py` and `dense_gemm_2.py` so readers see the same axis vocabulary, comments, and coordinate references down the tutorial track.
4. Run the modal example (`examples/blackwell/tutorial_gemm/*`) after the edits to confirm correctness.

## Constraints
- No L dimension should be introduced at the end of the local tiles; the shapes should stay `(tile_dim, Rest*, RestK/RestL)`.
- The change must not break the existing control flow, so the new shapes should index into fragments using explicit coordinates (e.g., `k_block_coord`).
- All edits should keep the tutorial narrative consistent—comments and variable names should feel like a natural evolution from part 1 to part 2.

## Proposed Changes
1. In `dense_gemm_1.py`, replace the `proj` argument when creating `gA`, `gB`, `gC` with full `(None, None, None)` coordinates plus the `mma_tiler` shape, so each tensor carries `(bM/bN, bK, Rest*, RestK, RestL)`.
2. Update any derived sizes (`cute.size(gA, mode=[3])`, etc.), partitions (`thr_mma.partition_*`), and any epilogue loops so they reference the new rest axes explicitly. Make sure the same coordinate math used elsewhere (e.g., `k_block_coord`) correctly indexes the added dimensions.
3. Review `dense_gemm_2.py` and rewrite comments/naming where necessary so it mirrors the terminology introduced in part 1 (e.g., consistently refer to `RestM`, `RestN`, `RestK`, `RestL`, and describe tensor shapes using the same parentheses notation).
4. Run the modal benchmark/tutorial for both scripts on the updated code to confirm they still pass reference checks.

## Testing
- Re-run the modal command (e.g., `python examples/blackwell/tutorial_gemm/dense_gemm_1.py --mnk ...`) for both tutorials after editing, verify the reference assertion still passes, and capture a summary of the result in the final report.

## Next Steps
1. After the spec is approved, draft a focused implementation plan (using the writing-plans skill) that lists the actual edits file-by-file.
2. Implement the shape/comment changes, update related fragments, and rerun tests.
3. Once both scripts are aligned, proceed to drafting the new tutorial narrative for `dense_gemm_2.py` with the clarified terminology.
