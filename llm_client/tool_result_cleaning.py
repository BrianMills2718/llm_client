"""Compatibility re-export. Canonical location: llm_client.tools.tool_result_cleaning."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.tools.tool_result_cleaning")
_sys.modules[__name__] = _canonical
