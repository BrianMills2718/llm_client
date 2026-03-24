"""Compatibility re-export. Canonical location: llm_client.core.model_detection."""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("llm_client.core.model_detection")
_sys.modules[__name__] = _canonical
