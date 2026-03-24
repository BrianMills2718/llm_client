"""Compatibility re-export. Canonical location: llm_client.utils.rate_limit."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.utils.rate_limit")
_sys.modules[__name__] = _canonical
