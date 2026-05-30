"""
Post-process pybind11-stubgen output for TensorRT bindings.

- Prefixes `bool` type hints with `builtins.bool` because TensorRT adds a global DataType boolean with the same name.
- Fixes parameter order syntax error.
- Renames parameters/docstrings named 'in' (reserved keyword) to 'in_'.
"""

import re
import sys
from pathlib import Path


def fix_file(file_path: Path) -> None:
    if not file_path.exists():
        print(f"Error: {file_path} does not exist.")
        return

    print(f"Fixing stubs in {file_path}...")
    content = file_path.read_text(encoding="utf-8")

    # Insert import builtins
    if "import builtins" not in content:
        if "from __future__ import annotations" in content:
            content = content.replace(
                "from __future__ import annotations", "from __future__ import annotations\nimport builtins"
            )
        else:
            content = "import builtins\n" + content

    # Fix syntax errors
    content = content.replace("channel_axis: typing.SupportsInt", "channel_axis: typing.SupportsInt = ...")
    content = re.sub(r"\bin\s*:", "in_:", content)

    # Replace 'bool' with 'builtins.bool' except inside quotes or the module-level 'bool: DataType'
    def replacer(match: re.Match[str]) -> str:
        start = match.start()
        end = match.end()

        # Check if surrounded by single or double quotes
        if (start > 0 and content[start - 1] in ("'", '"')) and (end < len(content) and content[end] in ("'", '"')):
            return "bool"

        # Check if it's the global attribute definition: "bool: DataType"
        # It starts at the beginning of a line (no indentation for global) and is followed by a colon.
        line_start = content.rfind("\n", 0, start) + 1
        prefix = content[line_start:start]
        suffix = content[end:]
        if prefix.strip() == "" and suffix.lstrip().startswith(":"):
            return "bool"

        return "builtins.bool"

    new_content = re.sub(r"\bbool\b", replacer, content)
    file_path.write_text(new_content, encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python fix_tensorrt_stubs.py <file1.pyi> <file2.pyi> ...")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        fix_file(Path(path_str))


if __name__ == "__main__":
    main()
