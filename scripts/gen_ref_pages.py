from __future__ import annotations

import inspect
from io import TextIOWrapper
from pathlib import Path
from pkgutil import iter_modules

import mkdocs_gen_files
import vsaa
import vsdeband
import vsdehalo
import vsdeinterlace
import vsdenoise
import vsexprtools
import vskernels
import vsmasktools
import vspreview
import vsrgtools
import vsscale
import vssource
import vstools
import vstransitions

# Modules to document.
MODULES = [
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
EXCLUDE = frozenset({
    # Cannot be found.
    "vspreview.plugins.builtins.frame_props",
    "vspreview.plugins.builtins.frame_props.category",
    "vspreview.plugins.builtins.frame_props.exclude",
    "vspreview.plugins.builtins.frame_props.lut",
    "vspreview.plugins.builtins.slowpics_comp",
    "vspreview.plugins.builtins.slowpics_comp.main",
    "vspreview.plugins.builtins.slowpics_comp.settings",
    "vspreview.plugins.builtins.slowpics_comp.utils",
    "vspreview.plugins.builtins.slowpics_comp.workers",

    # Cannot be found.
    "vstransitions.libs.movis",
})

# Explicitly included submodules that would otherwise not have been processed.
INCLUDE = frozenset({
    # Submodules are `_` prefixed, so include the overarching module.
    "vsmasktools.edge",
})

nav = mkdocs_gen_files.Nav()  # type: ignore[no-untyped-call]

for module in MODULES:
    src = Path(inspect.getfile(module)).parent
    site_packages = src.parent
    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(site_packages).with_suffix("")
        doc_path = path.relative_to(site_packages).with_suffix(".md")
        full_doc_path = Path("api", doc_path)
        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1].startswith("_") or ".".join(parts) in EXCLUDE:
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            assert isinstance(fd, TextIOWrapper)

            ident = ".".join(parts)

            if ident in EXCLUDE:
                continue

            # An `__init__.py`.
            if full_doc_path.name == "index.md":
                fd.writelines((
                    "---\n",
                    f"title: {ident}\n",
                    "---\n\n",
                    f"::: {ident}\n",
                    "    options:\n",
                    f"       members: {ident in INCLUDE}\n"
                ))

                # Top-level module (e.g. `vsaa`, `vsdeband`, etc.)
                if len(parts) == 1:
                    fd.write('<span class="doc-section-title">Submodules:</span>\n\n')
                    fd.writelines(
                        f"- [{sm.name}]({sm.name if not sm.ispkg else f'{sm.name}/index'}.md)\n"  # noqa: E501
                        for sm in iter_modules(module.__path__)
                        if not sm.name.startswith("_")
                    )
            else:
                fd.writelines((
                    "---\n",
                    f"title: {ident}\n",
                    "---\n\n",
                    f"::: {ident}\n",
                ))

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
