from .data_types import DictionaryType as DictionaryType
from .data_types import FloatTensorType as FloatTensorType
from .data_types import Int64TensorType as Int64TensorType
from .data_types import SequenceType as SequenceType
from .data_types import StringTensorType as StringTensorType
from .utils import check_input_and_output_numbers as check_input_and_output_numbers
from .utils import check_input_and_output_types as check_input_and_output_types

def calculate_linear_classifier_output_shapes(operator) -> None: ...
def calculate_linear_regressor_output_shapes(operator) -> None: ...
