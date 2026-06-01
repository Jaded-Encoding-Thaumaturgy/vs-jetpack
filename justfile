shebang := if os() == 'windows' {
  'pwsh.exe'
} else {
  '/usr/bin/env pwsh'
}
set windows-shell := ['pwsh.exe', '-CommandWithArgs']
set positional-arguments

stub-gen-trt:
  #!{{shebang}}
  uv run pybind11-stubgen tensorrt -o stubs
  uv run pybind11-stubgen tensorrt_bindings -o stubs `
    --enum-class-locations "PluginFieldType:tensorrt_bindings.tensorrt" `
    --enum-class-locations "EngineStat:tensorrt_bindings.tensorrt" `
    --enum-class-locations "ExecutionContextAllocationStrategy:tensorrt_bindings.tensorrt" `
    --enum-class-locations "Severity:tensorrt_bindings.tensorrt.ILogger" `
    --enum-class-locations "DataType:tensorrt_bindings.tensorrt" `
    --print-invalid-expressions-as-is
  uv run pybind11-stubgen tensorrt_rtx -o stubs
  uv run pybind11-stubgen tensorrt_rtx_bindings -o stubs `
    --enum-class-locations "PluginFieldType:tensorrt_rtx_bindings.tensorrt_rtx" `
    --enum-class-locations "EngineStat:tensorrt_rtx_bindings.tensorrt_rtx" `
    --enum-class-locations "ExecutionContextAllocationStrategy:tensorrt_rtx_bindings.tensorrt_rtx" `
    --enum-class-locations "Severity:tensorrt_rtx_bindings.tensorrt_rtx.ILogger" `
    --enum-class-locations "DataType:tensorrt_rtx_bindings.tensorrt_rtx" `
    --print-invalid-expressions-as-is
  uv run scripts/fix_tensorrt_stubs.py stubs/tensorrt_bindings/tensorrt.pyi stubs/tensorrt_rtx_bindings/tensorrt_rtx.pyi
  uv run ruff check --fix-only --unsafe-fixes --extend-fixable RUF013 PYI032 stubs/tensorrt stubs/tensorrt_bindings stubs/tensorrt_rtx stubs/tensorrt_rtx_bindings
  uv run ruff format stubs/tensorrt stubs/tensorrt_bindings stubs/tensorrt_rtx stubs/tensorrt_rtx_bindings
