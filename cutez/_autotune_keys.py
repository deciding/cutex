def resolve_autotune_key_values(kernel, *args, **kwargs):
    if hasattr(kernel, "autotune_key_values"):
        return kernel.autotune_key_values(*args, **kwargs)
    return {}
