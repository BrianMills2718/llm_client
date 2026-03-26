#!/usr/bin/env python3
"""Generate and verify the llm_client API reference docs.

The browser API reference is a generated projection of the package surface,
not a hand-maintained page. This script discovers public modules under
``llm_client``, extracts docstrings and typed signatures from live objects, and
renders:

* ``docs/API_REFERENCE.html`` for browser review
* ``docs/API_REFERENCE.md`` as a compact markdown index that links to the HTML

Run in write mode to regenerate the docs, or in check mode to fail if the
checked-in files drift from the generated output.
"""

from __future__ import annotations

import ast
import argparse
import html
import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "llm_client"
DEFAULT_HTML_PATH = REPO_ROOT / "docs" / "API_REFERENCE.html"
DEFAULT_MD_PATH = REPO_ROOT / "docs" / "API_REFERENCE.md"

ROOT_MODULE_NAME = PACKAGE_NAME
OPEN_MODULES = {
    "llm_client",
    "llm_client.client",
    "llm_client.io_log",
    "llm_client.models",
    "llm_client.prompts",
    "llm_client.observability",
}


@dataclass(frozen=True)
class SymbolDoc:
    """Docstring and signature data for a public function or method."""

    name: str
    signature: str
    summary: str
    doc: str
    source_module: str


@dataclass(frozen=True)
class ClassDoc:
    """Docstring and signature data for a public class and its methods."""

    name: str
    signature: str
    summary: str
    doc: str
    source_module: str
    methods: list[SymbolDoc] = field(default_factory=list)


@dataclass(frozen=True)
class ConstantDoc:
    """Rendered view of a public module constant."""

    name: str
    value_text: str
    value_type: str


@dataclass(frozen=True)
class ModuleDoc:
    """Rendered view of one importable package module."""

    name: str
    summary: str
    doc: str
    functions: list[SymbolDoc] = field(default_factory=list)
    classes: list[ClassDoc] = field(default_factory=list)
    constants: list[ConstantDoc] = field(default_factory=list)


def discover_module_names(package_name: str) -> list[str]:
    """Return all importable modules under ``package_name``.

    The package root is included first, followed by recursive descendants in
    depth order. Private modules and ``__main__`` are excluded because they are
    not part of the browsable library surface.
    """
    package = importlib.import_module(package_name)
    module_names = [package_name]
    prefix = f"{package_name}."
    for module_info in pkgutil.walk_packages(package.__path__, prefix=prefix):
        name = module_info.name
        leaf = name.rsplit(".", 1)[-1]
        if leaf.startswith("_") or leaf == "__main__":
            continue
        module_names.append(name)
    return sorted(module_names, key=lambda value: (value.count("."), value))


def load_module(name: str) -> ModuleType:
    """Import and return one package module."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(name)


def module_source_path(name: str) -> Path | None:
    """Return the source file path for one module if it can be located."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin in (None, "built-in", "namespace"):
        return None
    return Path(spec.origin)


def module_docstring_text(module: ModuleType) -> str:
    """Return the module docstring or a stable fallback string."""
    return inspect.getdoc(module) or "No module docstring available."


def first_paragraph(doc: str) -> str:
    """Return the first paragraph of a docstring for summary display."""
    paragraphs = [block.strip() for block in doc.split("\n\n") if block.strip()]
    if not paragraphs:
        return "No docstring available."
    return re.sub(r"\s+", " ", paragraphs[0]).strip()


def signature_text(obj: Any) -> str:
    """Render a callable or class signature with evaluated annotations."""
    try:
        sig = inspect.signature(obj, eval_str=True)
    except (TypeError, ValueError, NameError):
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return "(signature unavailable)"
    return str(sig)


def public_names(module: ModuleType) -> list[str]:
    """Return the public symbol names to document for one module."""
    exported = getattr(module, "__all__", None)
    if isinstance(exported, (list, tuple)):
        return [name for name in exported if hasattr(module, name)]

    names: list[str] = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismethod(value):
            if getattr(value, "__module__", module.__name__) == module.__name__:
                names.append(name)
            continue
        if name.isupper() and not inspect.ismodule(value):
            names.append(name)
    return names


def constant_value_text(value: Any) -> str:
    """Render a constant value in a deterministic, readable form."""
    rendered = stable_literal_text(value)
    if len(rendered) > 240:
        return f"<{type(value).__name__}>"
    return rendered


def stable_literal_text(value: Any) -> str:
    """Render common literal containers with deterministic ordering."""
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: stable_sort_key(item[0]))
        inner = ", ".join(
            f"{stable_literal_text(key)}: {stable_literal_text(item_value)}"
            for key, item_value in items
        )
        return f"{{{inner}}}"
    if isinstance(value, frozenset):
        inner = ", ".join(sorted(stable_literal_text(item) for item in value))
        return f"frozenset({{{inner}}})"
    if isinstance(value, set):
        inner = ", ".join(sorted(stable_literal_text(item) for item in value))
        return f"{{{inner}}}"
    if isinstance(value, tuple):
        inner = ", ".join(stable_literal_text(item) for item in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, list):
        inner = ", ".join(stable_literal_text(item) for item in value)
        return f"[{inner}]"
    return repr(value)


def stable_sort_key(value: Any) -> str:
    """Return a deterministic sort key for nested literal rendering."""
    return stable_literal_text(value)


def is_public_constant(name: str, value: Any) -> bool:
    """Return True when a module attribute should be rendered as a constant."""
    return name.isupper() and not inspect.isroutine(value) and not inspect.isclass(value)


def collect_symbol_doc(module: ModuleType, name: str, value: Any) -> SymbolDoc:
    """Collect documentation for one public function or method."""
    doc = inspect.getdoc(value) or ""
    return SymbolDoc(
        name=name,
        signature=signature_text(value),
        summary=first_paragraph(doc),
        doc=doc or "No docstring available.",
        source_module=getattr(value, "__module__", module.__name__),
    )


def collect_class_doc(module: ModuleType, name: str, value: Any) -> ClassDoc:
    """Collect documentation for one public class and its direct methods."""
    doc = inspect.getdoc(value) or ""
    methods: list[SymbolDoc] = []
    for attr_name, attr_value in value.__dict__.items():
        if attr_name.startswith("_"):
            continue
        if not callable(attr_value) and not isinstance(attr_value, (staticmethod, classmethod)):
            continue
        bound_value = getattr(value, attr_name)
        methods.append(collect_symbol_doc(module, attr_name, bound_value))

    methods.sort(key=lambda item: item.name)
    return ClassDoc(
        name=name,
        signature=signature_text(value),
        summary=first_paragraph(doc),
        doc=doc or "No docstring available.",
        source_module=getattr(value, "__module__", module.__name__),
        methods=methods,
    )


def collect_module_doc(name: str) -> ModuleDoc:
    """Collect the browsable documentation payload for one module."""
    try:
        module = load_module(name)
    except ModuleNotFoundError:
        source_path = module_source_path(name)
        if source_path is None or not source_path.exists():
            raise
        return collect_module_doc_from_source(name, source_path)
    return collect_module_doc_from_module(name, module)


def collect_module_doc_from_module(name: str, module: ModuleType) -> ModuleDoc:
    """Collect module docs from an imported module object."""
    doc = module_docstring_text(module)
    functions: list[SymbolDoc] = []
    classes: list[ClassDoc] = []
    constants: list[ConstantDoc] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for symbol_name in public_names(module):
            value = getattr(module, symbol_name)
            if inspect.isclass(value):
                classes.append(collect_class_doc(module, symbol_name, value))
            elif inspect.isroutine(value):
                functions.append(collect_symbol_doc(module, symbol_name, value))
            elif is_public_constant(symbol_name, value):
                constants.append(
                    ConstantDoc(
                        name=symbol_name,
                        value_text=constant_value_text(value),
                        value_type=type(value).__name__,
                    )
                )

    functions.sort(key=lambda item: item.name)
    classes.sort(key=lambda item: item.name)
    constants.sort(key=lambda item: item.name)
    return ModuleDoc(
        name=name,
        summary=first_paragraph(doc),
        doc=doc,
        functions=functions,
        classes=classes,
        constants=constants,
    )


def collect_module_doc_from_source(name: str, source_path: Path) -> ModuleDoc:
    """Collect module docs by parsing source when importing is not possible."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree) or "No module docstring available."
    functions: list[SymbolDoc] = []
    classes: list[ClassDoc] = []
    constants: list[ConstantDoc] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append(
                SymbolDoc(
                    name=node.name,
                    signature=ast_signature_text(node),
                    summary=first_paragraph(ast.get_docstring(node) or ""),
                    doc=ast.get_docstring(node) or "No docstring available.",
                    source_module=name,
                )
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(collect_class_doc_from_ast(name, node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and is_public_constant(target.id, object()):
                    constants.append(
                        ConstantDoc(
                            name=target.id,
                            value_text=ast.unparse(node.value),
                            value_type="literal",
                        )
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if is_public_constant(node.target.id, object()) and node.value is not None:
                constants.append(
                    ConstantDoc(
                        name=node.target.id,
                        value_text=ast.unparse(node.value),
                        value_type="literal",
                    )
                )

    functions.sort(key=lambda item: item.name)
    classes.sort(key=lambda item: item.name)
    constants.sort(key=lambda item: item.name)
    return ModuleDoc(
        name=name,
        summary=first_paragraph(doc),
        doc=doc,
        functions=functions,
        classes=classes,
        constants=constants,
    )


def ast_signature_text(node: ast.FunctionDef | ast.AsyncFunctionDef, *, drop_first_arg: bool = False) -> str:
    """Render a function signature from AST when runtime introspection is unavailable."""
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if drop_first_arg and positional:
        positional = positional[1:]
        if args.posonlyargs:
            posonly_count = max(0, len(args.posonlyargs) - 1)
            posonlyargs = positional[:posonly_count]
        else:
            posonlyargs = []
    else:
        posonlyargs = list(args.posonlyargs)

    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pieces: list[str] = []

    def render_arg(arg: ast.arg, default: ast.expr | None = None) -> str:
        text = arg.arg
        if arg.annotation is not None:
            text += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        return text

    for index, arg in enumerate(positional):
        default = defaults[index]
        pieces.append(render_arg(arg, default))
        if posonlyargs and index == len(posonlyargs) - 1:
            pieces.append("/")

    if args.vararg is not None:
        pieces.append(render_arg(args.vararg))
    elif args.kwonlyargs:
        pieces.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        pieces.append(render_arg(arg, default))

    if args.kwarg is not None:
        pieces.append(render_arg(args.kwarg, None).replace(args.kwarg.arg, f"**{args.kwarg.arg}", 1))

    rendered = ", ".join(piece for piece in pieces if piece)
    if node.returns is not None:
        return f"({rendered}) -> {ast.unparse(node.returns)}"
    return f"({rendered})"


def collect_class_doc_from_ast(module_name: str, node: ast.ClassDef) -> ClassDoc:
    """Collect class docs from parsed source when importing is not possible."""
    doc = ast.get_docstring(node) or "No docstring available."
    methods: list[SymbolDoc] = []
    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            init_node = item
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
            methods.append(
                SymbolDoc(
                    name=item.name,
                    signature=ast_signature_text(item),
                    summary=first_paragraph(ast.get_docstring(item) or ""),
                    doc=ast.get_docstring(item) or "No docstring available.",
                    source_module=module_name,
                )
            )

    methods.sort(key=lambda item: item.name)
    signature = "()"
    if init_node is not None:
        signature = ast_signature_text(init_node, drop_first_arg=True)
    return ClassDoc(
        name=node.name,
        signature=signature,
        summary=first_paragraph(doc),
        doc=doc,
        source_module=module_name,
        methods=methods,
    )


def module_anchor(name: str) -> str:
    """Convert a module name to a stable HTML anchor id."""
    return f"module-{name.replace('.', '-')}"


def symbol_anchor(module_name: str, symbol_name: str) -> str:
    """Convert a module and symbol name to a stable HTML anchor id."""
    return f"{module_anchor(module_name)}-{symbol_name.replace('.', '-')}"


def render_doc_block(doc: str) -> str:
    """Render a docstring as a readable HTML block."""
    return f'<pre class="doc">{html.escape(doc)}</pre>'


def render_symbol_details(module_name: str, symbol: SymbolDoc, kind: str) -> str:
    """Render one function or method as a collapsible HTML details block."""
    anchor = symbol_anchor(module_name, symbol.name)
    return f"""
<details class="symbol" id="{anchor}">
  <summary><code>{html.escape(symbol.name)}</code> <span class="sig">{html.escape(symbol.signature)}</span></summary>
  <div class="symbol-body">
    <p class="summary">{html.escape(symbol.summary)}</p>
    <p class="meta">Source: <code>{html.escape(symbol.source_module)}</code> {html.escape(kind)}</p>
    {render_doc_block(symbol.doc)}
  </div>
</details>
"""


def render_class_details(module_name: str, class_doc: ClassDoc) -> str:
    """Render one public class and its methods as a collapsible block."""
    anchor = symbol_anchor(module_name, class_doc.name)
    method_html = "".join(
        render_symbol_details(module_name, method, "method")
        for method in class_doc.methods
    )
    methods_section = ""
    if method_html:
        methods_section = f"""
    <h4>Methods</h4>
    {method_html}
    """
    return f"""
<details class="symbol class-symbol" id="{anchor}">
  <summary><code>{html.escape(class_doc.name)}</code> <span class="sig">{html.escape(class_doc.signature)}</span></summary>
  <div class="symbol-body">
    <p class="summary">{html.escape(class_doc.summary)}</p>
    <p class="meta">Source: <code>{html.escape(class_doc.source_module)}</code> class</p>
    {render_doc_block(class_doc.doc)}
    {methods_section}
  </div>
</details>
"""


def render_constants_table(constants: list[ConstantDoc]) -> str:
    """Render a module's constants as an HTML table."""
    if not constants:
        return ""
    rows = "\n".join(
        f"<tr><td><code>{html.escape(item.name)}</code></td>"
        f"<td><code>{html.escape(item.value_text)}</code></td>"
        f"<td>{html.escape(item.value_type)}</td></tr>"
        for item in constants
    )
    return f"""
<h4>Constants</h4>
<table>
  <thead>
    <tr><th>Name</th><th>Value</th><th>Type</th></tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""


def render_symbol_table(title: str, items: list[SymbolDoc], module_name: str) -> str:
    """Render a list of functions or methods for one module."""
    if not items:
        return ""
    rendered = []
    for item in items:
        rendered.append(render_symbol_details(module_name, item, title.lower()))
    return f"""
<h4>{html.escape(title)}</h4>
{''.join(rendered)}
"""


def render_class_table(module_name: str, classes: list[ClassDoc]) -> str:
    """Render a module's public classes."""
    if not classes:
        return ""
    rendered = "".join(render_class_details(module_name, item) for item in classes)
    return f"""
<h4>Classes</h4>
{rendered}
"""


def render_module_section(module_doc: ModuleDoc, *, open_section: bool = False) -> str:
    """Render one module as a collapsible documentation section."""
    module_id = module_anchor(module_doc.name)
    open_attr = " open" if open_section else ""
    body_parts = [f'<p class="summary">{html.escape(module_doc.summary)}</p>']
    if module_doc.doc:
        body_parts.append(render_doc_block(module_doc.doc))
    body_parts.append(render_symbol_table("Functions", module_doc.functions, module_doc.name))
    body_parts.append(render_class_table(module_doc.name, module_doc.classes))
    body_parts.append(render_constants_table(module_doc.constants))
    body_html = "\n".join(part for part in body_parts if part.strip())
    return f"""
<details class="module" id="{module_id}"{open_attr}>
  <summary><span class="module-name">{html.escape(module_doc.name)}</span> <span class="module-summary">{html.escape(module_doc.summary)}</span></summary>
  <div class="module-body">
    {body_html}
  </div>
</details>
"""


def render_root_surface(root_doc: ModuleDoc) -> str:
    """Render the package-root surface as a docs page section."""
    rows = []
    for symbol in root_doc.functions:
        rows.append(
            f"<tr><td><code>{html.escape(symbol.name)}</code></td>"
            f"<td class=\"sig\"><code>{html.escape(symbol.signature)}</code></td>"
            f"<td>{html.escape(symbol.summary)}</td></tr>"
        )
    for class_doc in root_doc.classes:
        rows.append(
            f"<tr><td><code>{html.escape(class_doc.name)}</code></td>"
            f"<td class=\"sig\"><code>{html.escape(class_doc.signature)}</code></td>"
            f"<td>{html.escape(class_doc.summary)}</td></tr>"
        )
    table_html = ""
    if rows:
        table_html = f"""
<table>
  <thead>
    <tr><th>Symbol</th><th>Signature</th><th>Summary</th></tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
"""
    constants_html = render_constants_table(root_doc.constants)
    return f"""
<section class="root-surface" id="root-surface">
  <h2>Package Root Surface</h2>
  <p class="summary">{html.escape(root_doc.summary)}</p>
  {render_doc_block(root_doc.doc)}
  {table_html}
  {constants_html}
</section>
"""


def render_markdown_index(root_doc: ModuleDoc, module_docs: list[ModuleDoc]) -> str:
    """Render the concise markdown index that points at the HTML docs."""
    from datetime import datetime, timezone
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# API Reference",
        f"<!-- Generated: {gen_ts} -->",
        "",
        "Generated from package docstrings and typed signatures.",
        "",
        "Browser view: [API_REFERENCE.html](API_REFERENCE.html)",
        "",
        "## Start Here",
        "",
        "1. [README.md](../README.md) for installation, usage, routing, and examples.",
        "2. [AGENTS.md](../AGENTS.md) for repo operating rules and architectural boundaries.",
        "3. [docs/plans/01_master-roadmap.md](plans/01_master-roadmap.md) for the current long-term program state.",
        "",
        "## Core Runtime Surface",
        "",
    ]

    for symbol in root_doc.functions[:8]:
        lines.append(f"- `{symbol.name}` - {symbol.summary}")
    if root_doc.classes:
        for class_doc in root_doc.classes[:4]:
            lines.append(f"- `{class_doc.name}` - {class_doc.summary}")

    lines.extend(
        [
            "",
            "## Module Catalog",
            "",
            f"Generated from {len(module_docs)} importable modules under `llm_client`.",
            "",
            "Open the HTML file for the full module-by-module docs surface.",
            "",
            "## Source Of Truth",
            "",
            "1. `pyproject.toml` is authoritative for package metadata and extras.",
            "2. Module docstrings and public function signatures are authoritative for code behavior.",
            "3. The roadmap and ADRs are authoritative for architectural boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(root_doc: ModuleDoc, module_docs: list[ModuleDoc]) -> str:
    """Render the full browser documentation page as HTML."""
    from datetime import datetime, timezone
    _html_gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    toc_items = [
        '<li><a href="#start-here">Start Here</a></li>',
        '<li><a href="#root-surface">Package Root Surface</a></li>',
        '<li><a href="#module-catalog">Module Catalog</a></li>',
        '<li><a href="#source-truth">Source Of Truth</a></li>',
    ]
    toc_items.extend(
        f'<li><a href="#{module_anchor(module_doc.name)}">{html.escape(module_doc.name)}</a></li>'
        for module_doc in module_docs
    )

    module_sections = []
    for module_doc in module_docs:
        module_sections.append(
            render_module_section(
                module_doc,
                open_section=module_doc.name in OPEN_MODULES,
            )
        )

    return f"""<!-- Generated: {_html_gen_ts} -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llm_client API Reference</title>
  <style>
    :root {{
      --fg: #1f2328;
      --muted: #57606a;
      --line: #d0d7de;
      --bg: #ffffff;
      --code: #f6f8fa;
      --link: #0969da;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--fg);
      font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 24px 72px;
    }}
    header {{
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    h1, h2, h3, h4 {{
      line-height: 1.2;
      margin: 0 0 12px;
      font-weight: 600;
    }}
    h1 {{ font-size: 2rem; }}
    h2 {{
      font-size: 1.25rem;
      margin-top: 34px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }}
    h4 {{
      margin-top: 18px;
      font-size: 1rem;
    }}
    p {{ margin: 0 0 12px; }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    ul, ol {{ margin: 8px 0 12px 24px; }}
    li {{ margin: 4px 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f6f8fa;
      font-weight: 600;
    }}
    code {{
      background: var(--code);
      border: 1px solid #ebedf0;
      border-radius: 6px;
      padding: 1px 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
      word-break: break-word;
    }}
    pre {{
      background: var(--code);
      border: 1px solid #ebedf0;
      border-radius: 8px;
      padding: 12px 14px;
      overflow: auto;
      margin: 12px 0 18px;
      font-size: 14px;
      line-height: 1.5;
    }}
    details.module {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      margin: 12px 0;
      background: #fff;
    }}
    details.module > summary {{
      cursor: pointer;
      list-style: none;
      font-weight: 600;
    }}
    details.symbol {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      margin: 10px 0;
      background: #fff;
    }}
    details.symbol > summary {{
      cursor: pointer;
      list-style: none;
      font-weight: 600;
    }}
    .module-name {{
      margin-right: 10px;
    }}
    .module-summary, .summary, .meta {{
      color: var(--muted);
      font-weight: 400;
    }}
    .sig {{
      color: var(--fg);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      font-weight: 400;
    }}
    .doc {{
      white-space: pre-wrap;
    }}
    .toc {{
      background: #fafbfc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 18px 0 6px;
      max-height: 28rem;
      overflow: auto;
    }}
    .toc strong {{
      display: block;
      margin-bottom: 6px;
    }}
    hr {{
      border: 0;
      border-top: 1px solid var(--line);
      margin: 28px 0;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>llm_client API Reference</h1>
      <p class="meta">Generated from live package docstrings and type signatures.</p>
      <p>This page is the browser-openable docs surface for the package. It is produced by a generator script and should not be hand-edited.</p>
    </header>

    <section class="toc" aria-label="Table of contents">
      <strong>Contents</strong>
      <ol>
        {''.join(toc_items)}
      </ol>
    </section>

    <h2 id="start-here">Start Here</h2>
    <ol>
      <li><a href="{html.escape((REPO_ROOT / 'README.md').as_uri())}">README.md</a> for installation, usage, routing, and examples.</li>
      <li><a href="{html.escape((REPO_ROOT / 'AGENTS.md').as_uri())}">AGENTS.md</a> for repo operating rules and architectural boundaries.</li>
      <li><a href="{html.escape((REPO_ROOT / 'docs' / 'plans' / '01_master-roadmap.md').as_uri())}">docs/plans/01_master-roadmap.md</a> for the current long-term program state.</li>
      <li><a href="{html.escape((REPO_ROOT / 'docs' / 'ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md').as_uri())}">docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md</a> for the shared ecosystem boundary model.</li>
    </ol>

    {render_root_surface(root_doc)}

    <section id="module-catalog">
      <h2>Module Catalog</h2>
      <p class="summary">Each module section is generated from the live package surface. Public functions, classes, methods, and constants are listed from the module object itself.</p>
      {''.join(module_sections)}
    </section>

    <h2 id="source-truth">Source Of Truth</h2>
    <ol>
      <li><a href="{html.escape((REPO_ROOT / 'pyproject.toml').as_uri())}">pyproject.toml</a> is authoritative for package metadata and extras.</li>
      <li>Module docstrings and public function signatures are authoritative for code behavior.</li>
      <li>The roadmap and ADRs are authoritative for architectural boundaries.</li>
    </ol>
    <p class="summary">Canonical markdown companion: <a href="{html.escape((REPO_ROOT / 'docs' / 'API_REFERENCE.md').as_uri())}">docs/API_REFERENCE.md</a>.</p>
  </main>
</body>
</html>
"""


def generate_documents(package_name: str) -> tuple[str, str]:
    """Generate the markdown and HTML reference content for one package."""
    module_names = discover_module_names(package_name)
    module_docs = [collect_module_doc(name) for name in module_names]
    root_doc = module_docs[0]
    html_text = render_html(root_doc, module_docs[1:])
    markdown_text = render_markdown_index(root_doc, module_docs)
    return markdown_text, html_text


def write_text(path: Path, content: str) -> None:
    """Write text to a file, creating the parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def compare_or_write(*, markdown_path: Path, html_path: Path, markdown_text: str, html_text: str, check: bool) -> int:
    """Write generated docs or verify that the checked-in files are current."""
    if check:
        problems: list[str] = []
        if not markdown_path.exists():
            problems.append(f"missing markdown output: {markdown_path}")
        elif markdown_path.read_text(encoding="utf-8") != markdown_text:
            problems.append(f"stale markdown output: {markdown_path}")
        if not html_path.exists():
            problems.append(f"missing html output: {html_path}")
        elif html_path.read_text(encoding="utf-8") != html_text:
            problems.append(f"stale html output: {html_path}")
        if problems:
            print("API reference docs are out of sync:")
            for problem in problems:
                print(f"- {problem}")
            print("Run: python scripts/meta/generate_api_reference.py --write")
            return 1
        print("API reference docs are in sync.")
        return 0

    write_text(markdown_path, markdown_text)
    write_text(html_path, html_text)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {html_path}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate or check the llm_client API reference docs.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify checked-in docs match the generated output instead of writing files.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the docs on disk (default when --check is not set).",
    )
    parser.add_argument(
        "--package",
        default=PACKAGE_NAME,
        help="Package name to document (default: llm_client).",
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="Output markdown index path.",
    )
    parser.add_argument(
        "--html-path",
        type=Path,
        default=DEFAULT_HTML_PATH,
        help="Output HTML docs path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the API reference generator."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    markdown_text, html_text = generate_documents(args.package)
    return compare_or_write(
        markdown_path=args.markdown_path,
        html_path=args.html_path,
        markdown_text=markdown_text,
        html_text=html_text,
        check=args.check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
