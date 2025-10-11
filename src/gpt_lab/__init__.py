# Initialize multi-root catalog namespace paths at import time.
try:
    from .catalog_bootstrap import bootstrap_namespace_paths
    bootstrap_namespace_paths()
except Exception:
    # Avoid hard failures on import; discovery tools may still work partially.
    pass


