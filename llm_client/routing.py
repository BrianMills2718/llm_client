"""Compatibility re-export. Canonical location: llm_client.core.routing."""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("llm_client.core.routing")
_sys.modules[__name__] = _canonical
