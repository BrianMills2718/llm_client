"""Compatibility re-export. Canonical location: llm_client.execution.call_wrappers."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.execution.call_wrappers")
_sys.modules[__name__] = _canonical
