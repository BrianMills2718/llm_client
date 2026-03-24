"""Compatibility re-export. Canonical location: llm_client.agent.compliance_gate."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.agent.compliance_gate")
_sys.modules[__name__] = _canonical
