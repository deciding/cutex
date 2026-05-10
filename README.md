# cutex

## Install
```
pip install cutez
```

## Usage
```python
# Step 1: define the grid search scope
AUTOTUNE_CONFIGS = [
    cutez.Config(kwargs={"mma_tiler_mn": (256, 256), "cluster_shape_mn": (2, 1), "ab_stages": ab_stages,})
    for ab_stages in (6, 7, 8)
]

...

# Step 2: decorate the host function. can specify the persistent cache path
@cutez.autotune(configs=AUTOTUNE_CONFIGS, key=["m", "n", "k"],  cache_path='/workspace/dump/dense_gemm_7min.json')
@cute.jit
def host_function(
...


# Step 3: autotune requires compile with cutez.compile. use verbose=True to see the configs.
compiled_gemm = cutez.compile(
    ...
    verbose=True, # To show what is the best config
)

```

## Build Wheel

Build the source distribution and wheel with:

```bash
python -m build --sdist --wheel
```
