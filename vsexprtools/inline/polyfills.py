import builtins
import ctypes
import math
from functools import wraps
from typing import Any

from vstools import copy_func

from .helpers import ExprVar, op

__all__ = ["disable_poly", "enable_poly"]

# ruff: noqa

global _to_patch

dunder_methods = [
    "__add__",
    "__iadd__",
    "__sub__",
    "__isub__",
    "__mul__",
    "__imul__",
    "__floordiv__",
    "__ifloordiv__",
    "__pow__",
    "__exp__",
    "__log__",
    "__sqrt__",
    "__neg__",
    "__pos__",
    "__invert__",
    "__int__",
    "__float__",
    "__abs__",
    "__mod__",
    "__and__",
    "__or__",
    "__xor__",
]

ob_types = (float,)

builtin_methods = {
    (ob_type, dunder): copy_func(getattr(ob_type, dunder))
    for dunder in dunder_methods
    for ob_type in ob_types
    if dunder in dir(ob_type)
}

Py_ssize_t = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32

tp_as_dict = dict[tuple[Any, Any], Any]()
tp_func_dict = dict[tuple[Any, Any], Any]()


class PyObject(ctypes.Structure):
    def incref(self) -> None:
        self.ob_refcnt += 1

    def decref(self) -> None:
        self.ob_refcnt -= 1


class PyFile(ctypes.Structure):
    pass


PyObject_p = ctypes.py_object
Inquiry_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p)
# return type is void* to allow ctypes to convert python integers to
# plain PyObject*
UnaryFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p)
BinaryFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p, PyObject_p)
TernaryFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p, PyObject_p, PyObject_p)
LenFunc_p = ctypes.CFUNCTYPE(Py_ssize_t, PyObject_p)
SSizeArgFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p, Py_ssize_t)
SSizeObjArgProc_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, Py_ssize_t, PyObject_p)
ObjObjProc_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, PyObject_p)

FILE_p = ctypes.POINTER(PyFile)


def get_not_implemented() -> Any:
    namespace = dict[Any, Any]()
    name = "_Py_NotImplmented"
    not_implemented = ctypes.cast(ctypes.pythonapi._Py_NotImplementedStruct, ctypes.py_object)

    ctypes.pythonapi.PyDict_SetItem(ctypes.py_object(namespace), ctypes.py_object(name), not_implemented)
    return namespace[name]


# address of the _Py_NotImplementedStruct singleton
NotImplementedRet = get_not_implemented()


class PyNumberMethods(ctypes.Structure):
    _fields_ = [
        ("nb_add", BinaryFunc_p),
        ("nb_subtract", BinaryFunc_p),
        ("nb_multiply", BinaryFunc_p),
        ("nb_remainder", BinaryFunc_p),
        ("nb_divmod", BinaryFunc_p),
        ("nb_power", BinaryFunc_p),
        ("nb_negative", UnaryFunc_p),
        ("nb_positive", UnaryFunc_p),
        ("nb_absolute", UnaryFunc_p),
        ("nb_bool", Inquiry_p),
        ("nb_invert", UnaryFunc_p),
        ("nb_lshift", BinaryFunc_p),
        ("nb_rshift", BinaryFunc_p),
        ("nb_and", BinaryFunc_p),
        ("nb_xor", BinaryFunc_p),
        ("nb_or", BinaryFunc_p),
        ("nb_int", UnaryFunc_p),
        ("nb_reserved", ctypes.c_void_p),
        ("nb_float", UnaryFunc_p),
        ("nb_inplace_add", BinaryFunc_p),
        ("nb_inplace_subtract", BinaryFunc_p),
        ("nb_inplace_multiply", BinaryFunc_p),
        ("nb_inplace_remainder", BinaryFunc_p),
        ("nb_inplace_power", TernaryFunc_p),
        ("nb_inplace_lshift", BinaryFunc_p),
        ("nb_inplace_rshift", BinaryFunc_p),
        ("nb_inplace_and", BinaryFunc_p),
        ("nb_inplace_xor", BinaryFunc_p),
        ("nb_inplace_or", BinaryFunc_p),
        ("nb_floor_divide", BinaryFunc_p),
        ("nb_true_divide", BinaryFunc_p),
        ("nb_inplace_floor_divide", BinaryFunc_p),
        ("nb_inplace_true_divide", BinaryFunc_p),
        ("nb_index", BinaryFunc_p),
        ("nb_matrix_multiply", BinaryFunc_p),
        ("nb_inplace_matrix_multiply", BinaryFunc_p),
    ]


class PySequenceMethods(ctypes.Structure):
    _fields_ = [
        ("sq_length", LenFunc_p),
        ("sq_concat", BinaryFunc_p),
        ("sq_repeat", SSizeArgFunc_p),
        ("sq_item", SSizeArgFunc_p),
        ("was_sq_slice", ctypes.c_void_p),
        ("sq_ass_item", SSizeObjArgProc_p),
        ("was_sq_ass_slice", ctypes.c_void_p),
        ("sq_contains", ObjObjProc_p),
        ("sq_inplace_concat", BinaryFunc_p),
        ("sq_inplace_repeat", SSizeArgFunc_p),
    ]


class PyMappingMethods(ctypes.Structure):
    pass


class PyTypeObject(ctypes.Structure):
    pass


class PyAsyncMethods(ctypes.Structure):
    pass


PyObject._fields_ = [
    ("ob_refcnt", Py_ssize_t),
    ("ob_type", ctypes.POINTER(PyTypeObject)),
]


PyTypeObject._fields_ = [
    # varhead
    ("ob_base", PyObject),
    ("ob_size", Py_ssize_t),
    # declaration
    ("tp_name", ctypes.c_char_p),
    ("tp_basicsize", Py_ssize_t),
    ("tp_itemsize", Py_ssize_t),
    ("tp_dealloc", ctypes.CFUNCTYPE(None, PyObject_p)),
    ("printfunc", ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, FILE_p, ctypes.c_int)),
    ("getattrfunc", ctypes.CFUNCTYPE(PyObject_p, PyObject_p, ctypes.c_char_p)),
    ("setattrfunc", ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, ctypes.c_char_p, PyObject_p)),
    ("tp_as_async", ctypes.CFUNCTYPE(PyAsyncMethods)),
    ("tp_repr", ctypes.CFUNCTYPE(PyObject_p, PyObject_p)),
    ("tp_as_number", ctypes.POINTER(PyNumberMethods)),
    ("tp_as_sequence", ctypes.POINTER(PySequenceMethods)),
    ("tp_as_mapping", ctypes.POINTER(PyMappingMethods)),
    ("tp_hash", ctypes.CFUNCTYPE(ctypes.c_int64, PyObject_p)),
    ("tp_call", ctypes.CFUNCTYPE(PyObject_p, PyObject_p, PyObject_p, PyObject_p)),
    ("tp_str", ctypes.CFUNCTYPE(PyObject_p, PyObject_p)),
]


PyTypeObject_as_types_dict: dict[str, type[ctypes.Structure]] = {
    "tp_as_async": PyAsyncMethods,
    "tp_as_number": PyNumberMethods,
    "tp_as_sequence": PySequenceMethods,
    "tp_as_mapping": PyMappingMethods,
}

# build override infomation for dunder methods
as_number = (
    "tp_as_number",
    [
        ("add", "nb_add"),
        ("sub", "nb_subtract"),
        ("mul", "nb_multiply"),
        ("mod", "nb_remainder"),
        ("pow", "nb_power"),
        ("neg", "nb_negative"),
        ("pos", "nb_positive"),
        ("abs", "nb_absolute"),
        ("bool", "nb_bool"),
        ("inv", "nb_invert"),
        ("invert", "nb_invert"),
        ("lshift", "nb_lshift"),
        ("rshift", "nb_rshift"),
        ("and", "nb_and"),
        ("xor", "nb_xor"),
        ("or", "nb_or"),
        ("int", "nb_int"),
        ("float", "nb_float"),
        ("iadd", "nb_inplace_add"),
        ("isub", "nb_inplace_subtract"),
        ("imul", "nb_inplace_multiply"),
        ("imod", "nb_inplace_remainder"),
        ("ipow", "nb_inplace_power"),
        ("ilshift", "nb_inplace_lshift"),
        ("irshift", "nb_inplace_rshift"),
        ("iadd", "nb_inplace_and"),
        ("ixor", "nb_inplace_xor"),
        ("ior", "nb_inplace_or"),
        ("floordiv", "nb_floor_divide"),
        ("div", "nb_true_divide"),
        ("ifloordiv", "nb_inplace_floor_divide"),
        ("idiv", "nb_inplace_true_divide"),
        ("index", "nb_index"),
        ("matmul", "nb_matrix_multiply"),
        ("imatmul", "nb_inplace_matrix_multiply"),
    ],
)

override_dict = {}
tp_as_name = as_number[0]
for dunder, impl_method in as_number[1]:
    override_dict["__{}__".format(dunder)] = (tp_as_name, impl_method)

override_dict["divmod()"] = ("tp_as_number", "nb_divmod")
override_dict["__str__"] = ("tp_str", "tp_str")

_to_patch = False


def _curse(klass: Any, attr: Any, func: Any) -> None:
    assert callable(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            if _to_patch:
                return func(*map(_try_cast(klass), args), **kwargs)
            return func(*args, **kwargs)
        except NotImplementedError:
            return NotImplementedRet

    tp_as_name, impl_method = override_dict[attr]

    tyobj = PyTypeObject.from_address(id(klass))

    if tp_as_name in PyTypeObject_as_types_dict:
        struct_ty = PyTypeObject_as_types_dict[tp_as_name]
        tp_as_ptr = getattr(tyobj, tp_as_name)

        if not tp_as_ptr:
            tp_as_obj = struct_ty()
            tp_as_dict[(klass, attr)] = tp_as_obj
            tp_as_new_ptr = ctypes.cast(ctypes.addressof(tp_as_obj), ctypes.POINTER(struct_ty))

            setattr(tyobj, tp_as_name, tp_as_new_ptr)
        tp_as = tp_as_ptr[0]

        for fname, ftype in struct_ty._fields_:  # type: ignore[misc]
            if fname == impl_method:
                cfunc_t = ftype

        cfunc = cfunc_t(wrapper)
        tp_func_dict[(klass, attr)] = cfunc

        setattr(tp_as, impl_method, cfunc)
    else:
        for fname, ftype in PyTypeObject._fields_:  # type: ignore[misc]
            if fname == impl_method:
                cfunc_t = ftype

        if (klass, attr) not in tp_as_dict:
            tp_as_dict[(klass, attr)] = ctypes.cast(getattr(tyobj, impl_method), cfunc_t)  # type: ignore[type-var]

        cfunc = cfunc_t(wrapper)
        tp_func_dict[(klass, attr)] = cfunc
        setattr(tyobj, impl_method, cfunc)


def _reverse(klass: Any, attr: Any) -> None:
    tp_as_name, impl_method = override_dict[attr]
    tyobj = PyTypeObject.from_address(id(klass))
    tp_as_ptr = getattr(tyobj, tp_as_name)
    if tp_as_ptr:
        if tp_as_name in PyTypeObject_as_types_dict:
            tp_as = tp_as_ptr[0]

            struct_ty = PyTypeObject_as_types_dict[tp_as_name]
            for fname, ftype in struct_ty._fields_:  # type: ignore[misc]
                if fname == impl_method:
                    cfunc_t = ftype

            setattr(tp_as, impl_method, ctypes.cast(ctypes.c_void_p(None), cfunc_t))  # type: ignore[type-var]
        else:
            if (klass, attr) not in tp_as_dict:
                return

            cfunc = tp_as_dict[(klass, attr)]
            setattr(tyobj, impl_method, cfunc)


def _try_cast(klass: Any) -> Any:
    def e(v: Any) -> Any:
        try:
            return klass(v)
        except BaseException:
            return v

    return e


def _poly(op: Any, k: Any) -> Any:
    def inner(*args: Any, **kwargs: Any) -> Any:
        if not any(isinstance(x, ExprVar) for x in args):
            return substitutions[k]["min"][0](*args, **kwargs)

        var = args[0]
        for arg in args[1:]:
            var = op(var, arg)

        return var

    return inner


_builtins = {
    "min": (copy_func(builtins.min), _poly(op.min, "builtins")),
    "max": (copy_func(builtins.max), _poly(op.max, "builtins")),
}

_math = {"log": (copy_func(math.log), _poly(op.log, "math"))}


substitutions = {"builtins": _builtins, "math": _math}


def enable_poly() -> None:
    """
    Internal function to enable builtin methods override.
    """
    global _to_patch

    for k, v in substitutions.items():
        eval(k).__dict__.update(**{k: v[1] for k, v in v.items()})

    for obtype, dunder in builtin_methods:
        _curse(obtype, dunder, getattr(ExprVar, dunder))

    _to_patch = False


def disable_poly() -> None:
    """
    Internal function to disable builtin methods override.
    """
    global _to_patch

    for k, v in substitutions.items():
        eval(k).__dict__.update(**{k: v[0] for k, v in v.items()})

    for (obtype, dunder), dunfunc in builtin_methods.items():
        _curse(obtype, dunder, dunfunc)

    _to_patch = True
