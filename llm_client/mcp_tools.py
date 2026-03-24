"""Compatibility re-export. Canonical location: llm_client.agent.mcp_tools."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.agent.mcp_tools")
_sys.modules[__name__] = _canonical
