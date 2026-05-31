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
    """Bridge TensorRT log messages into a standard Python logger."""

    def __init__(self, logger: logging.Logger) -> None:
        """
        Args:
            logger: Destination logger for TensorRT messages.
        """
        super().__init__()
        self.logger = logger

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        """
        Forward a TensorRT log callback to Python logging.

        Args:
            severity: TensorRT log severity.
            msg: Message emitted by TensorRT.
        """
        level = SEVERITY_MAP[severity]
        self.logger.log(level, msg, exc_info=level >= logging.ERROR, stacklevel=2)
