from . import utils as utils
from .metadata_props import add_metadata_props as add_metadata_props
from _typeshed import Incomplete

DEFAULT_OPSET_NUMBER: int
OPSET_TO_IR_VERSION: Incomplete

def onnx_builtin_opset_version(): ...
def get_maximum_opset_supported(): ...
def make_model_ex(graph, imported_opset_pairs, target_default_opset, metadata_props=None, **kwargs): ...
