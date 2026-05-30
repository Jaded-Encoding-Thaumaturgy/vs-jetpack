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
  --enum-class-locations "PluginFieldType:tensorrt_bindings.tensorrt.PluginFieldType" `
  --enum-class-locations "EngineStat:tensorrt_bindings.tensorrt.EngineStat" `
  --enum-class-locations "ExecutionContextAllocationStrategy:tensorrt_bindings.tensorrt.ExecutionContextAllocationStrategy" `
  --enum-class-locations "Severity:tensorrt_bindings.tensorrt.ILogger.Severity" `
  --enum-class-locations "DataType:tensorrt_bindings.tensorrt.DataType" `
  --ignore-invalid-expressions "(nvinfer1|nvonnxparser|detail)::.*"
  uv run pybind11-stubgen tensorrt_rtx -o stubs
  uv run pybind11-stubgen tensorrt_rtx_bindings -o stubs `
  --enum-class-locations "PluginFieldType:tensorrt_rtx_bindings.tensorrt_rtx.PluginFieldType" `
  --enum-class-locations "EngineStat:tensorrt_rtx_bindings.tensorrt_rtx.EngineStat" `
  --enum-class-locations "ExecutionContextAllocationStrategy:tensorrt_rtx_bindings.tensorrt_rtx.ExecutionContextAllocationStrategy" `
  --enum-class-locations "Severity:tensorrt_rtx_bindings.tensorrt_rtx.ILogger.Severity" `
  --enum-class-locations "DataType:tensorrt_rtx_bindings.tensorrt_rtx.DataType" `
  --ignore-invalid-expressions "(nvinfer1|nvonnxparser|detail)::.*"
  uv run scripts/fix_tensorrt_stubs.py stubs/tensorrt_bindings/tensorrt.pyi stubs/tensorrt_rtx_bindings/tensorrt_rtx.pyi
  uv run ruff format stubs/tensorrt stubs/tensorrt_bindings stubs/tensorrt_rtx stubs/tensorrt_rtx_bindings
  uv run ruff check --fix-only --ignore E501 --ignore PYI021 stubs/tensorrt stubs/tensorrt_bindings stubs/tensorrt_rtx stubs/tensorrt_rtx_bindings
