"""Compatibility re-export. Canonical location: llm_client.sdk.agents."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.sdk.agents")
_sys.modules[__name__] = _canonical
