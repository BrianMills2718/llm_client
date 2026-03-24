"""Compatibility re-export. Canonical location: llm_client.execution.background_runtime."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.execution.background_runtime")
_sys.modules[__name__] = _canonical
