import logging

import tensorrt as trt

SEVERITY_MAP = {
    trt.ILogger.Severity.VERBOSE: logging.DEBUG,
    trt.ILogger.Severity.INFO: logging.INFO,
    trt.ILogger.Severity.WARNING: logging.WARNING,
    trt.ILogger.Severity.ERROR: logging.ERROR,
    trt.ILogger.Severity.INTERNAL_ERROR: logging.CRITICAL,
}


class Logger(trt.ILogger):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        level = SEVERITY_MAP[severity]
        self.logger.log(level, msg, exc_info=level >= logging.ERROR, stacklevel=2)
