from __future__ import annotations

import inspect
from collections.abc import Iterable
from io import TextIOWrapper
from pathlib import Path
from pkgutil import iter_modules
from typing import Any

import jetpytools
import mkdocs_gen_files
import vspreview
import vstransitions
from orderedsets import FrozenOrderedSet

import vsaa
import vsdeband
import vsdehalo
import vsdeinterlace
import vsdenoise
import vsexprtools
import vskernels
import vsmasktools
import vsrgtools
import vsscale
import vssource
import vstools

# Modules to document.
MODULES = [
    jetpytools,
    vsaa,
    vsdeband,
    vsdehalo,
    vsdeinterlace,
    vsdenoise,
    vsexprtools,
    vskernels,
    vsmasktools,
    vspreview,
    vsrgtools,
    vsscale,
    vssource,
    vstools,
    vstransitions,
]

# Excluded submodules.
EXCLUDE = [FrozenOrderedSet(path.split(".")) for path in {"vspreview"}]

# Explicitly included submodules that would otherwise not have been processed.
# vsmasktools.edge submodules are `_` prefixed, so include the overarching module.
INCLUDE = [FrozenOrderedSet(path.split(".")) for path in {"vsmasktools.edge", "vspreview.api"}]


def is_excluded(s: Iterable[Any]) -> bool:
    return any(excl.issubset(s) for excl in EXCLUDE)


def is_included(s: Iterable[Any]) -> bool:
    return any(excl.issubset(s) for excl in INCLUDE)


nav = mkdocs_gen_files.Nav()

for module in MODULES:
    src = Path(inspect.getfile(module)).parent

    site_packages = src.parent

    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(site_packages).with_suffix("")
        doc_path = path.relative_to(site_packages).with_suffix(".md")
        full_doc_path = Path("api", doc_path)
        parts = tuple(module_path.parts)

        if is_included(parts):
            pass
        elif is_excluded(parts):
            continue

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1].startswith("_"):
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            assert isinstance(fd, TextIOWrapper)

            ident = ".".join(parts)

            if is_excluded(parts) and not is_included(parts):
                continue

            # An `__init__.py`.
            if full_doc_path.name == "index.md":
                fd.writelines(
                    (
                        "---\n",
                        f"title: {ident}\n",
                        "---\n\n",
                        f"::: {ident}\n",
                        "    options:\n",
                        f"       members: {is_included(parts) and not is_excluded(parts)}\n",
                    )
                )

                # Top-level module (e.g. `vsaa`, `vsdeband`, etc.)
                if len(parts) == 1:
                    fd.write('<span class="doc-section-title">Submodules:</span>\n\n')
                    fd.writelines(
                        f"- [{sm.name}]({sm.name if not sm.ispkg else f'{sm.name}/index'}.md)\n"
                        for sm in iter_modules(module.__path__)
                        if not sm.name.startswith("_")
                    )
            else:
                fd.writelines(
                    (
                        "---\n",
                        f"title: {ident}\n",
                        "---\n\n",
                        f"::: {ident}\n",
                    )
                )

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
