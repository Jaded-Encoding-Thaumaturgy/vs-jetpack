# Contributing

## Development Installation

**[uv](https://github.com/astral-sh/uv)** is the default tool used for development in this repository.

Clone the repository and sync all packages:

```bash
git clone https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack.git
cd vs-jetpack
uv sync --all-extras
```

Install and enable the local git hooks:

```bash
uv run prek install
```

## Recommended Editor Settings

### VSCode / VSCodium

```json
{
  "[github-actions-workflow]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports.ruff": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  },
  "[toml]": {
    "editor.defaultFormatter": "tamasfe.even-better-toml",
    "editor.formatOnSave": true
  },
  "autoDocstring.docstringFormat": "google-notypes",
  "editor.formatOnSave": true,
  "mypy-type-checker.args": ["--fixed-format-cache"],
  "mypy-type-checker.importStrategy": "fromEnvironment",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.enableTroubleshootMissingImports": true,
  "python.analysis.extraPaths": ["./"],
  "python.analysis.importFormat": "absolute",
  "python.analysis.includeAliasesFromUserFiles": true,
  "python.analysis.showOnlyDirectDependenciesInAutoImport": false,
  "python.analysis.stubPath": "stubs",
  "python.analysis.typeCheckingMode": "standard",
  "python.analysis.typeEvaluation.deprecateTypingAliases": true,
  "python.testing.pytestArgs": ["tests"],
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```
