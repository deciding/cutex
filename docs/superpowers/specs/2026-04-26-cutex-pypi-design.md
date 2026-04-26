# Cutex PyPI Packaging Design

## Goal

Publish this repository as a PyPI package named `cutex` that depends on `flash_attn_4==4.0.0b4` and exposes a minimal forwarding namespace for selected `cutlass` APIs under `cutex`.

This first cut should keep the package surface intentionally small and should not modify or interfere with the upstream `cutlass` package.

## Scope

This design covers only the initial packaging and namespace exposure required to make `cutex` installable from PyPI and usable as a thin compatibility namespace.

Included in scope:
- Add Python packaging metadata for a PyPI-publishable package
- Add a local `cutex` package
- Re-export `cutlass.cute` public names directly from `cutex`
- Re-export selected `cutlass.cute` related modules and symbols directly from `cutex`
- Re-export `cutlass.pipeline` through `cutex.pipeline`
- Re-export `cutlass.utils` through `cutex.utils`
- Re-export `cutlass.utils.blackwell_helpers` through `cutex.utils.sm100`
- Add tests for the promised public import surface

Out of scope for this first cut:
- Re-exporting flash-attn-4 methods under `cutex`
- Creating a full mirrored package tree such as `cutex.cute.runtime`
- Preserving or altering `cutlass.*` import behavior
- Refactoring existing repo modules to import `cutex`

## Package Layout

The initial package should contain exactly these new top-level packaging files and package modules:

- `pyproject.toml`
- `cutex/__init__.py`
- `cutex/pipeline.py`
- `cutex/utils/__init__.py`
- `cutex/utils/sm100.py`

Test files will also be added under the repository's chosen test location.

## Dependency Model

The package should declare `flash_attn_4==4.0.0b4` as an install dependency.

`cutex` should not vendor `cutlass` code and should not patch any upstream modules. It should rely on the installed dependency to provide:
- `cutlass.cute`
- `cutlass.cute.nvgpu`
- `cutlass.cute.runtime`
- `cutlass.cute.testing`
- `cutlass.pipeline`
- `cutlass.utils`
- `cutlass.utils.blackwell_helpers`

If those upstream imports are unavailable, imports from `cutex` should fail naturally with standard Python import errors.

## Public API Design

### `cutex`

`cutex/__init__.py` is the main public namespace.

It should:
- Execute `from cutlass.cute import *`
- Import and expose `nvgpu` as a top-level module attribute on `cutex`
- Import and expose `runtime` as a top-level module attribute on `cutex`
- Import and expose `testing` as a top-level module attribute on `cutex`
- Import and expose `cpasync` as a top-level attribute on `cutex`
- Import and expose `tcgen05` as a top-level attribute on `cutex`
- Import and expose `from_dlpack` as a top-level attribute on `cutex`

The intended usage is:

```python
import cutex

cutex.cpasync
cutex.tcgen05
cutex.from_dlpack
cutex.nvgpu
cutex.runtime
cutex.testing
```

And because of the wildcard re-export, representative public names from `cutlass.cute` should also be reachable directly as `cutex.<name>`.

### `cutex.pipeline`

`cutex/pipeline.py` should be a forwarding module for `cutlass.pipeline`.

The intended usage is:

```python
from cutex.pipeline import pipeline_init_arrive, pipeline_init_wait
```

### `cutex.utils`

`cutex/utils/__init__.py` should be a forwarding module for `cutlass.utils`.

The intended usage is:

```python
import cutex.utils as utils
```

### `cutex.utils.sm100`

`cutex/utils/sm100.py` should be a forwarding module for `cutlass.utils.blackwell_helpers`.

The intended usage is:

```python
import cutex.utils.sm100 as sm100_utils
```

## Implementation Notes

The implementation should stay thin. The package files should act as forwarding layers only.

Preferred behavior:
- Keep module code minimal and declarative
- Avoid custom import hook machinery
- Avoid compatibility aliases for `cutlass`
- Avoid adding broader namespace mirroring than required by the files listed above

Because `cutex` is meant to feel like `cute` at the top level, `cutex/__init__.py` should prioritize direct symbol exposure rather than requiring a nested `cutex.cute` path.

## Testing Strategy

Tests should cover only the promised import surface for this first cut.

Required test coverage:
- `import cutex`
- `from cutex import cpasync, tcgen05, from_dlpack`
- `hasattr(cutex, "nvgpu")`
- `hasattr(cutex, "runtime")`
- `hasattr(cutex, "testing")`
- Representative verification that names re-exported from `cutlass.cute` are directly available on `cutex`
- `from cutex.pipeline import pipeline_init_arrive, pipeline_init_wait`
- `import cutex.utils as utils`
- `import cutex.utils.sm100 as sm100_utils`

TDD requirement:
- Write the import tests first
- Run them and confirm they fail because the package does not yet exist
- Add the minimal forwarding implementation
- Re-run tests to confirm the package surface works

## Validation

Before considering the work complete, verify:
- The package can be discovered by Python packaging tools
- The declared dependency includes `flash_attn_4==4.0.0b4`
- The import tests pass against the intended public surface
- No changes alter or monkey-patch the upstream `cutlass` package

## Future Extension

Later work can extend `cutex` to re-export flash-attn-4 methods under the same namespace. That work should be designed separately so this first packaging cut remains minimal and low-risk.
