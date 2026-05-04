# `compute_epilogue_tile_shape` Analysis

This note analyzes `compute_epilogue_tile_shape` in:

- `cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py`
- function starting at line `82`

In this repo, the helper is currently coming from the installed CUTLASS DSL package at:

- `/home/zining/miniconda3/lib/python3.13/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/blackwell_helpers.py`

## Purpose

`compute_epilogue_tile_shape` chooses the 2D tile shape used by the epilogue store path, especially the TMEM-load/TMA-store epilogue flow on SM100.

In `modal/blackwell/dense_gemm_7min.py`, it is used like this:

```python
epi_tile = sm100_utils.compute_epilogue_tile_shape(
    cta_tile_shape_mnk,
    use_2cta_instrs,
    c_layout,
    io_dtype,
)
```

and then passed into:

```python
c_smem_layout_staged = sm100_utils.make_smem_layout_epi(...)
tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(..., epi_tile)
```

So the function is answering:

- how large the epilogue tile should be in `M` and `N`
- so that the epilogue store path is legal and reasonably efficient
- given CTA tile shape, datatype widths, layout, and whether the kernel uses 2-CTA instructions

## Inputs

The function signature is:

```python
def compute_epilogue_tile_shape(
    cta_tile_shape: cute.Shape,
    use_2cta_instrs: bool,
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    *,
    layout_c: LayoutEnum = None,
    elem_ty_c: Union[Type[Numeric], None] = None,
    loc=None,
    ip=None,
) -> cute.Tile:
```

Important inputs:

- `cta_tile_shape[:2]`
  - CTA output tile `(cta_m, cta_n)`
- `use_2cta_instrs`
  - whether the kernel is using the 2-CTA instruction mode
- `layout_d`
  - layout of output tensor `D`
- `elem_ty_d`
  - output element type
- optional `layout_c`, `elem_ty_c`
  - source tensor `C` layout and element type for source-using epilogues

If `elem_ty_c is None`, the function treats the epilogue as source-disabled.

## Step-By-Step Logic

### 1. Validate types

The helper checks that `elem_ty_d` and optional `elem_ty_c` are CUTLASS numeric types.

```python
def validate_type(ty, ty_name):
    if not isinstance(ty, NumericMeta):
        raise TypeError(...)
```

This is just defensive validation.

### 2. Extract CTA tile dimensions

```python
cta_m, cta_n = cta_tile_shape[:2]
```

Only the `M` and `N` dimensions matter for the epilogue tile decision.

### 3. Choose warp arrangement heuristic

```python
(warp_m, warp_n) = (2, 2) if (cta_m == 64 and use_2cta_instrs) else (4, 1)
```

This is a key policy choice.

Normally the epilogue assumes a `4 x 1` warp grouping.

For the special case:

- `cta_m == 64`
- `use_2cta_instrs == True`

it switches to `2 x 2`.

This directly affects:

- how large the epilogue tile can be in `M`
- what the minimum/legal `N` granularity must be
- how the returned tile is factored across warp groups

### 4. Detect whether the epilogue uses a source tensor

```python
disable_source = elem_ty_c == None
```

If `True`, the epilogue is effectively output-only. If `False`, it must satisfy both source and destination constraints.

### 5. Compute the widest participating type

```python
max_bits = (
    elem_ty_d.width if disable_source else max(elem_ty_c.width, elem_ty_d.width)
)
```

This width influences how many elements can be processed efficiently.

### 6. Choose `tile_m`

```python
dp_full = 32
tile_m = min(cta_m, dp_full * warp_m)
```

Interpretation:

- normal case: `warp_m = 4`, so `tile_m = min(cta_m, 128)`
- 2CTA special case: `warp_m = 2`, so `tile_m = min(cta_m, 64)`

This caps epilogue tile height even if the CTA tile is taller.

### 7. Choose a performance-oriented `N` target: `n_perf`

If source is disabled:

```python
if max_bits == 4:
    compute_elts = 8192
else:
    compute_elts = 4096
n_perf = compute_elts // tile_m
```

If source is enabled:

```python
if max_bits == 32:
    n_perf = 16 if (cta_m > 64 and cta_n <= 128) else 32
elif max_bits == 16:
    n_perf = 32 if cta_n <= 128 else 64
else:
    n_perf = 64
```

This section is heuristic tuning logic rather than strict legality. It tries to pick a tile width in `N` that is large enough to perform well for the datatype/layout mix.

### 8. Determine layout major-ness

```python
d_is_m_major = layout_d.is_m_major_c()
c_is_m_major = True if layout_c is None else layout_c.is_m_major_c()
```

This matters because `M`-major outputs are easier to store with a smaller `N` requirement, while non-`M`-major layouts need stronger vectorization/alignment constraints.

### 9. Compute minimum legal-ish `N` constraints

For output `D`:

```python
n_min_d = (
    8 * warp_n
    if d_is_m_major
    else (128 * warp_n if elem_ty_d.width == 6 else 128 // elem_ty_d.width * warp_n)
)
```

For source `C`:

```python
n_min_c = (
    8 * warp_n
    if (c_is_m_major or disable_source)
    else (128 * warp_n if elem_ty_c.width == 6 else 128 // elem_ty_c.width * warp_n)
)
```

Interpretation:

- `M`-major layouts can tolerate a relatively small minimum `N`: `8 * warp_n`
- non-`M`-major layouts need a larger `N` depending on element width
- narrower types need more elements to reach the same bit-level transfer granularity
- width-6 is treated specially because packing is awkward

This is the main legality/alignment logic of the helper.

### 10. Choose final `tile_n`

```python
tile_n = min(cta_n, max(n_perf, n_min_c, n_min_d))
```

So final `tile_n` is:

- never larger than CTA `N`
- but large enough to satisfy:
  - performance heuristic target `n_perf`
  - source-side minimum `n_min_c`
  - destination-side minimum `n_min_d`

### 11. Reject unsupported CTA widths

```python
if cta_n < n_min_c or cta_n < n_min_d:
    raise ValueError(f"CTA tile too small: {cta_tile_shape=}")
```

This means the CTA tile itself must already be wide enough in `N`. If it is too narrow, the epilogue path is unsupported for the chosen layout/type combination.

### 12. Return a layout-structured tile, not a plain tuple

```python
tile_m_layout = cute.make_layout(tile_m, loc=loc, ip=ip)
tile_n_layout = cute.make_layout(
    (tile_n // warp_n, warp_n), stride=(1, cta_n // warp_n), loc=loc, ip=ip
)
return (tile_m_layout, cute.coalesce(tile_n_layout, loc=loc, ip=ip))
```

This is an important detail.

The function does **not** return:

```python
(tile_m, tile_n)
```

It returns:

- an `M` layout object
- an `N` layout object factored by `warp_n`

So the result carries not only size, but also a specific arrangement that downstream partitioning expects.

## Why `N` Is Factored By `warp_n`

This line is the key:

```python
(tile_n // warp_n, warp_n)
```

If `warp_n = 2`, then `N` is split into:

- outer chunks
- an inner `2`-way warp grouping

Then `cute.coalesce(...)` simplifies the layout while preserving the intended grouping.

This makes the epilogue tile line up with warp-level ownership and with the epilogue store path’s expected thread/value structure.

## What It Means In `dense_gemm_7min.py`

Your call site is:

```python
epi_tile = sm100_utils.compute_epilogue_tile_shape(
    cta_tile_shape_mnk,
    use_2cta_instrs,
    c_layout,
    io_dtype,
)
```

Notice that it does **not** pass `layout_c` or `elem_ty_c`.

So in this kernel:

- source is disabled
- the helper is sizing the epilogue tile only for the `D` store path
- `n_perf` comes from the source-disabled branch:

```python
compute_elts = 4096 or 8192
n_perf = compute_elts // tile_m
```

- legality checks come only from `D` and the no-source path

That means for `dense_gemm_7min.py`, the helper is mostly deciding how much of the CTA result should be handled as one epilogue tile while matching SM100 epilogue store constraints and warp grouping.

## Main Conceptual Summary

`compute_epilogue_tile_shape` is a heuristic plus constraint solver for the epilogue tile.

It balances:

- CTA tile size
- 1CTA vs 2CTA epilogue warp arrangement
- output type width
- optional source type width
- output and source layout major-ness
- minimum legal vectorization widths
- a preferred performance-oriented width in `N`

and returns a layout-structured epilogue tile suitable for:

- epilogue shared-memory layout construction
- TMA store atom creation
- downstream epilogue thread/value partitioning

## Most Important Lines

If you are scanning the function quickly, these are the most meaningful points:

- line `125`: choose warp arrangement heuristic
- line `132`: choose `tile_m`
- lines `134-146`: choose performance-oriented `n_perf`
- lines `151-160`: compute minimum `N` constraints from layout/type
- line `161`: choose final `tile_n`
- lines `163-164`: reject unsupported CTA widths
- lines `167-171`: return layout-structured epilogue tile

## Short Summary In One Sentence

`compute_epilogue_tile_shape` picks an epilogue tile that is both legal and performant for the Blackwell epilogue store path, then returns it as a CuTe layout-structured tile rather than a plain `(m, n)` shape.
