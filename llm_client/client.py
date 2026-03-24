"""Compatibility re-export. Canonical location: llm_client.core.client."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.core.client")
_sys.modules[__name__] = _canonical
