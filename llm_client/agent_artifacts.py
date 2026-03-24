"""Compatibility re-export. Canonical location: llm_client.agent.agent_artifacts."""
import importlib as _importlib
import sys as _sys
_canonical = _importlib.import_module("llm_client.agent.agent_artifacts")
_sys.modules[__name__] = _canonical
