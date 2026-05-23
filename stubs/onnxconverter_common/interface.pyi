import abc

class ModelContainer(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def add_initializer(self, name, onnx_type, shape, content): ...
    @abc.abstractmethod
    def add_node(self, op_type, inputs, outputs, op_domain: str = '', op_version: int = 1, **attrs): ...

class OperatorBase(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta
    @property
    @abc.abstractmethod
    def full_name(self): ...
    @property
    @abc.abstractmethod
    def input_full_names(self): ...
    @property
    @abc.abstractmethod
    def output_full_names(self): ...
    @property
    @abc.abstractmethod
    def original_operator(self): ...

class ScopeBase:
    __metaclass__ = abc.ABCMeta
