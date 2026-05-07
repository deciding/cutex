from time import perf_counter_ns


def benchmark(fn, *args, warmup=0, rep=0, **kwargs):
    if not callable(fn):
        return fn

    for _ in range(warmup):
        fn(*args, **kwargs)

    best = None
    for _ in range(rep or 1):
        start = perf_counter_ns()
        fn(*args, **kwargs)
        elapsed_ns = perf_counter_ns() - start
        if best is None or elapsed_ns < best:
            best = elapsed_ns

    return best
