def resolve_autotune_key_values(kernel, *args, **kwargs):
    values = {}
    if hasattr(kernel, "autotune_init_kwargs"):
        values.update(kernel.autotune_init_kwargs())
    if hasattr(kernel, "autotune_key_values"):
        values.update(kernel.autotune_key_values(*args, **kwargs))
    return values
