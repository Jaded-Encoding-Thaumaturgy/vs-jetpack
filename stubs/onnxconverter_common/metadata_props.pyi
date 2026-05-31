from _typeshed import Incomplete

from .case_insensitive_dict import CaseInsensitiveDict as CaseInsensitiveDict

KNOWN_METADATA_PROPS: Incomplete

def add_metadata_props(onnx_model, metadata_props, target_opset) -> None: ...
def set_denotation(onnx_model, input_name, denotation, target_opset, dimension_denotation=None): ...
