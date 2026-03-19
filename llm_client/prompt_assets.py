"""Explicit prompt asset manifests and deterministic prompt_ref resolution.

This module implements the smallest shared prompt asset layer for ``llm_client``.
Prompt assets are addressed by explicit references such as
``shared.summarize.concise@1``. Resolution is deterministic:

1. parse the reference into ``id`` and ``version``,
2. map that reference to one manifest path under the package asset root,
3. validate the manifest against a strict schema,
4. resolve the template path relative to the manifest directory,
5. fail loudly if any part of the contract is missing or inconsistent.

There is intentionally no override chain, shadowing, or search-order magic.
Callers must provide either an explicit prompt asset reference or an explicit
filesystem path to a prompt template.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

_PROMPT_REF_RE = re.compile(r"^(?P<asset_id>[a-z0-9][a-z0-9_.-]*)@(?P<version>[1-9][0-9]*)$")
_PROMPT_ASSET_ROOT = Path(__file__).resolve().parent / "prompt_assets"


class PromptAssetRef(BaseModel):
    """Parsed prompt asset identity.

    The parsed reference is used to derive the canonical on-disk manifest
    location. The dotted ``asset_id`` becomes nested directories and ``version``
    becomes a ``vN`` directory segment.
    """

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(description="Stable prompt asset identifier without the version suffix.")
    version: int = Field(description="Positive integer version number.")

    @property
    def prompt_ref(self) -> str:
        """Return the canonical ``id@version`` reference string."""

        return f"{self.asset_id}@{self.version}"

    def manifest_path(self, asset_root: Path | None = None) -> Path:
        """Return the deterministic manifest path for this reference.

        Args:
            asset_root: Optional asset root override. Defaults to the built-in
                package prompt asset root.

        Returns:
            Absolute path to the manifest file for this prompt asset reference.
        """

        root = asset_root or _PROMPT_ASSET_ROOT
        return root.joinpath(*self.asset_id.split("."), f"v{self.version}", "manifest.yaml")


class PromptAssetManifest(BaseModel):
    """Strict metadata contract for a shared prompt asset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable prompt asset identifier.")
    version: int = Field(description="Positive integer version for this asset.")
    kind: Literal["chat_prompt"] = Field(
        default="chat_prompt",
        description="Prompt asset kind. Only chat prompt templates are supported today.",
    )
    status: Literal["draft", "candidate", "canonical", "deprecated"] = Field(
        description="Lifecycle state for the asset.",
    )
    description: str | None = Field(
        default=None,
        description="Short explanation of the prompt asset's intended use.",
    )
    derived_from: str | None = Field(
        default=None,
        description="Explicit parent prompt reference when this asset derives from another asset.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable prompt asset tags.",
    )
    template_path: str = Field(
        description="Prompt template path relative to the manifest directory.",
    )
    input_schema: str | None = Field(
        default=None,
        description="Optional schema identifier for render inputs.",
    )
    output_schema: str | None = Field(
        default=None,
        description="Optional schema identifier for expected prompt outputs.",
    )


class ResolvedPromptAsset(BaseModel):
    """A fully resolved prompt asset ready for rendering."""

    model_config = ConfigDict(extra="forbid")

    prompt_ref: str = Field(description="Canonical prompt asset reference.")
    manifest: PromptAssetManifest = Field(description="Validated prompt asset manifest.")
    manifest_path: Path = Field(description="Absolute manifest path on disk.")
    template_path: Path = Field(description="Absolute prompt template path on disk.")


def parse_prompt_ref(prompt_ref: str) -> PromptAssetRef:
    """Parse and validate a prompt asset reference.

    Args:
        prompt_ref: Explicit prompt asset reference such as
            ``shared.summarize.concise@1``.

    Returns:
        Parsed prompt asset identity.

    Raises:
        ValueError: If the reference is malformed.
    """

    match = _PROMPT_REF_RE.fullmatch(prompt_ref)
    if match is None:
        raise ValueError(
            "Invalid prompt_ref format. Expected '<asset_id>@<positive_integer_version>', "
            f"got: {prompt_ref!r}"
        )
    return PromptAssetRef(
        asset_id=match.group("asset_id"),
        version=int(match.group("version")),
    )


def resolve_prompt_asset(
    prompt_ref: str,
    *,
    asset_root: Path | None = None,
) -> ResolvedPromptAsset:
    """Resolve a prompt asset reference into validated manifest and template paths.

    Args:
        prompt_ref: Explicit prompt asset reference to resolve.
        asset_root: Optional asset root override for tests or controlled local
            development. Defaults to the package prompt asset root.

    Returns:
        Fully resolved prompt asset metadata and paths.

    Raises:
        FileNotFoundError: If the manifest or template file does not exist.
        ValueError: If the manifest is malformed or inconsistent with the
            provided reference.
        yaml.YAMLError: If the manifest YAML is malformed.
    """

    parsed_ref = parse_prompt_ref(prompt_ref)
    manifest_path = parsed_ref.manifest_path(asset_root=asset_root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prompt asset manifest not found for {prompt_ref}: {manifest_path}"
        )

    raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError(
            f"Prompt asset manifest must be a mapping, got {type(raw_manifest).__name__}: "
            f"{manifest_path}"
        )

    manifest = PromptAssetManifest.model_validate(raw_manifest)
    if manifest.id != parsed_ref.asset_id or manifest.version != parsed_ref.version:
        raise ValueError(
            "Prompt asset manifest identity mismatch. "
            f"Expected {parsed_ref.prompt_ref}, found {manifest.id}@{manifest.version}: "
            f"{manifest_path}"
        )
    if manifest.derived_from is not None:
        parse_prompt_ref(manifest.derived_from)

    template_path = (manifest_path.parent / manifest.template_path).resolve()
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt asset template not found for {prompt_ref}: {template_path}"
        )

    return ResolvedPromptAsset(
        prompt_ref=parsed_ref.prompt_ref,
        manifest=manifest,
        manifest_path=manifest_path.resolve(),
        template_path=template_path,
    )


def load_prompt_asset(
    prompt_ref: str,
    *,
    asset_root: Path | None = None,
) -> PromptAssetManifest:
    """Load validated prompt asset metadata for an explicit prompt reference.

    Args:
        prompt_ref: Explicit prompt asset reference to load.
        asset_root: Optional asset root override for tests or controlled local
            development.

    Returns:
        Validated prompt asset manifest metadata.
    """

    return resolve_prompt_asset(prompt_ref, asset_root=asset_root).manifest
