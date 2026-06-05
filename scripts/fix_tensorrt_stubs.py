"""
Post-process pybind11-stubgen output for TensorRT bindings.

- Prefixes `bool` type hints with `builtins.bool` because TensorRT adds a global DataType boolean with the same name.
- Rewrites TensorRT C++ type names from docstring signatures to Python stub names.
- Fixes parameter order syntax error.
- Renames parameters/docstrings named 'in' (reserved keyword) to 'in_'.
"""

import re
import sys
from pathlib import Path

CXX_TYPE_REPLACEMENTS = {
    "detail::accessor<detail::accessor_policies::str_attr>": "IPluginRegistry",
    "nvinfer1::IBuilder": "Builder",
    "nvonnxparser::IParserError": "ParserError",
}

CXX_NAMESPACE_PATTERN = re.compile(r"\b(?:nvinfer1|nvonnxparser)::(?:v_\d+_\d+::)?([A-Za-z_][A-Za-z_0-9]*)\b")
RAW_CPP_PATTERN = re.compile(r"\b(?:nvinfer1|nvonnxparser|detail)::")
INVALID_ELLIPSIS_RETURN_PATTERN = re.compile(r"^\s*def .*->\s*\.\.\.\s*:", re.MULTILINE)


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
    content = content.replace("builtins.builtins.bool", "builtins.bool")
    content = content.replace("channel_axis: typing.SupportsInt", "channel_axis: typing.SupportsInt = ...")
    content = re.sub(r"\bin\s*:", "in_:", content)
    content = fix_cpp_docstring_types(content)

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

        if start >= len("builtins.") and content[start - len("builtins.") : start] == "builtins.":
            return "bool"

        return "builtins.bool"

    new_content = re.sub(r"\bbool\b", replacer, content)
    validate_content(file_path, new_content)
    file_path.write_text(new_content, encoding="utf-8")


def fix_cpp_docstring_types(content: str) -> str:
    for cpp_name, python_name in CXX_TYPE_REPLACEMENTS.items():
        content = content.replace(cpp_name, python_name)

    return CXX_NAMESPACE_PATTERN.sub(r"\1", content)


def validate_content(file_path: Path, content: str) -> None:
    errors = []
    if match := RAW_CPP_PATTERN.search(content):
        errors.append(f"raw C++ expression near offset {match.start()}: {match.group(0)}")
    if match := INVALID_ELLIPSIS_RETURN_PATTERN.search(content):
        errors.append(f"invalid ellipsis return annotation near offset {match.start()}")
    if "builtins.builtins.bool" in content:
        errors.append("duplicate builtins bool annotation")

    if errors:
        formatted_errors = "\n  - ".join(errors)
        raise ValueError(f"{file_path} still contains unsupported stub output:\n  - {formatted_errors}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python fix_tensorrt_stubs.py <file1.pyi> <file2.pyi> ...")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        fix_file(Path(path_str))


if __name__ == "__main__":
    main()
