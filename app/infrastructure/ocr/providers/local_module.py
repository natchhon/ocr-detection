import importlib.util
import sys
from pathlib import Path


def load_local_module(name: str, path: Path, package: bool = False):
    """Load explicitly selected local source, never a similarly named installed package."""
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise RuntimeError("A different model source is already loaded; restart the worker")
        return existing
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load local model module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module
