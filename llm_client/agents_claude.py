"""Compatibility re-export. Canonical location: llm_client.sdk.agents_claude."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.sdk.agents_claude")
_sys.modules[__name__] = _canonical
