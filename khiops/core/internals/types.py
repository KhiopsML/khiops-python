######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Khiops task argument mini-type system"""
from abc import ABC, abstractmethod

from khiops.core.internals.common import (
    is_dict_like,
    is_list_like,
    is_string_like,
    type_error_message,
)


class KhiopsTaskArgumentType(ABC):
    """An argument for a Khiops task

    It provides the services to:

    - Verify that an argument belongs to the type
    - Transform the argument to an "scenario language" argument

    This class is not instantiable.
    """

    def __init__(self):
        """See class docstring"""
        raise ValueError("KhiopsTaskArgumentType is not instantiable")

    @classmethod
    @abstractmethod
    def is_of_this_type(cls, arg):
        """Test if the argument belongs to this type"""

    @classmethod
    def check(cls, arg, arg_name):
        """Raises TypeError if the argument does not belongs to this type"""
        if not cls.is_of_this_type(arg):
            raise TypeError(type_error_message(arg_name, arg, cls.short_name()))

    @classmethod
    @abstractmethod
    def short_name(cls):
        """Returns a short name for this type"""

    @classmethod
    @abstractmethod
    def to_scenario_arg(cls, arg):
        """Returns a string representation to be used in an scenario file"""


class BoolType(KhiopsTaskArgumentType):
    """Boolean argument type"""

    @classmethod
    def is_of_this_type(cls, arg):
        return isinstance(arg, bool)

    @classmethod
    def short_name(cls):
        return "bool"

    @classmethod
    def to_scenario_arg(cls, arg):
        if arg:
            return "true"
        else:
            return "false"


class IntType(KhiopsTaskArgumentType):
    """Integer argument type"""

    @classmethod
    def is_of_this_type(cls, arg):
        return isinstance(arg, int)

    @classmethod
    def short_name(cls):
        return "int"

    @classmethod
    def to_scenario_arg(cls, arg):
        return str(arg)


class FloatType(KhiopsTaskArgumentType):
    """Float argument type

    Note that literal int's are valid float in this mini-type system. We accept int's
    because it isn't an error for a user to set a float argument to 2 instead of 2.0.
    """

    @classmethod
    def is_of_this_type(cls, arg):
        return isinstance(arg, (int, float))

    @classmethod
    def short_name(cls):
        return "float"

    @classmethod
    def to_scenario_arg(cls, arg):
        return str(arg)


class StringLikeType(KhiopsTaskArgumentType):
    """String like argument type

    The string-like type is defined as the union of ``str`` and ``bytes``.
    """

    @classmethod
    def is_of_this_type(cls, arg):
        return is_string_like(arg)

    @classmethod
    def short_name(cls):
        return "str"

    @classmethod
    def to_scenario_arg(cls, arg):
        return arg


class AbstractListType(KhiopsTaskArgumentType):
    """Base class for ListType containers

    See the factory method `ListType`.
    """

    registry = {}

    @classmethod
    def is_registered(cls, value_type):
        return value_type in cls.registry

    @classmethod
    def register_subclass(cls, subclass):
        cls.registry[subclass.get_value_type()] = subclass

    @classmethod
    def get_subclass(cls, value_type):
        return cls.registry[value_type]

    @classmethod
    @abstractmethod
    def get_value_type(cls):
        """Return the type of the list values"""

    @classmethod
    def is_of_this_type(cls, arg):
        arg_type_ok = is_list_like(arg)
        value_type = cls.get_value_type()
        if arg_type_ok:
            for value in arg:
                arg_type_ok = arg_type_ok and value_type.is_of_this_type(value)
                if not arg_type_ok:
                    break
        return arg_type_ok

    @classmethod
    def short_name(cls):
        return f"list-{cls.get_value_type().short_name()}"

    @classmethod
    def to_scenario_arg(cls, arg):
        scenario_arg = []
        value_type = cls.get_value_type()
        for value in arg:
            scenario_arg.append(value_type.to_scenario_arg(value))
        return scenario_arg


def ListType(value_type):  # pylint: disable=invalid-name
    """ListType factory method

    Lists are themselves of type ``list`` and they may contain a variable number of
    elements of a single type.

    Parameters
    ----------
    value_type : `KhiopsTaskArgumentType`
        The type for the values contained in the list.

    Returns
    -------
    type
        A class which inherits from `AbstractListType`.
    """

    # Check the type of the list values
    if not issubclass(value_type, KhiopsTaskArgumentType):
        raise TypeError(
            type_error_message(value_type, "value_type", KhiopsTaskArgumentType)
        )
    if not is_simple_type(value_type) and not issubclass(value_type, AbstractTupleType):
        raise TypeError(
            f"List value type must be simple or tuple. It is '{value_type.__name__}'"
        )

    # Return an already registered class or register and return fresh new one
    # pylint: disable=missing-class-docstring
    if AbstractListType.is_registered(value_type):
        list_type = AbstractListType.get_subclass(value_type)
    else:

        class ConcreteListType(AbstractListType):
            @classmethod
            def get_value_type(cls):
                return value_type

        list_type = ConcreteListType
        AbstractListType.register_subclass(list_type)

    return list_type


class AbstractDictType(KhiopsTaskArgumentType):
    """Base class for DictType containers

    See the factory method `DictType`.
    """

    registry = {}

    @classmethod
    def is_registered(cls, key_type, value_type):
        return (key_type, value_type) in cls.registry

    @classmethod
    def register_subclass(cls, subclass):
        cls.registry[subclass.get_key_type(), subclass.get_value_type()] = subclass

    @classmethod
    def get_subclass(cls, key_type, value_type):
        return cls.registry[(key_type, value_type)]

    @classmethod
    @abstractmethod
    def get_key_type(cls):
        """Returns the type of the dictionary keys"""

    @classmethod
    @abstractmethod
    def get_value_type(cls):
        """Return the type of the dictionary values"""

    @classmethod
    def is_of_this_type(cls, arg):
        arg_type_ok = is_dict_like(arg)
        key_type = cls.get_key_type()
        value_type = cls.get_value_type()
        if arg_type_ok:
            for key, value in arg.items():
                arg_type_ok = (
                    arg_type_ok
                    and key_type.is_of_this_type(key)
                    and value_type.is_of_this_type(value)
                )
                if not arg_type_ok:
                    break
        return arg_type_ok

    @classmethod
    def short_name(cls):
        return (
            "dict-"
            + f"{cls.get_key_type().short_name()}-{cls.get_value_type().short_name()}"
        )

    @classmethod
    def to_scenario_arg(cls, arg):
        scenario_arg = {}
        key_type = cls.get_key_type()
        value_type = cls.get_value_type()
        for key, value in arg.items():
            scenario_arg[key_type.to_scenario_arg(key)] = value_type.to_scenario_arg(
                value
            )
        return scenario_arg


def DictType(key_type, value_type):  # pylint: disable=invalid-name
    """DictType factory method

    Dicts are themselves of type ``dict`` and they contain key-value relations with
    fixed key and value types.

    Parameters
    ----------
    key_type : `KhiopsTaskArgumentType`
        Type of the dictionary's keys.
    value_type : `KhiopsTaskArgumentType`
        Type of the dictionary's values.

    Returns
    -------
    type
        A class which inherits from `AbstractDictType`.
    """
    # Check the type of the dictionary's key and value
    if not issubclass(key_type, KhiopsTaskArgumentType):
        raise TypeError(
            type_error_message(key_type, "key_type", KhiopsTaskArgumentType)
        )
    if not is_simple_type(key_type):
        raise TypeError(f"Dict key type must be simple. It is '{value_type.__name__}'")
    if not issubclass(value_type, KhiopsTaskArgumentType):
        raise TypeError(
            type_error_message(value_type, "value_type", KhiopsTaskArgumentType)
        )
    if not is_simple_type(value_type):
        raise TypeError(
            f"Dict value type must be simple. It is '{value_type.__name__}'"
        )

    # Return an already registered class or register and return fresh new one
    # pylint: disable=missing-class-docstring
    if AbstractDictType.is_registered(key_type, value_type):
        dict_type = AbstractDictType.get_subclass(key_type, value_type)
    else:

        class ConcreteDictType(AbstractDictType):
            @classmethod
            def get_key_type(cls):
                return key_type

            @classmethod
            def get_value_type(cls):
                return value_type

        dict_type = ConcreteDictType
        AbstractDictType.register_subclass(dict_type)

    return dict_type


class AbstractTupleType(KhiopsTaskArgumentType):
    """Base class for TupleTypes

    See the factory method `TupleType`.
    """

    registry = {}

    @classmethod
    def is_registered(cls, tuple_types):
        return tuple_types in cls.registry

    @classmethod
    def register_subclass(cls, subclass):
        cls.registry[subclass.get_value_types()] = subclass

    @classmethod
    def get_subclass(cls, tuple_types):
        return cls.registry[tuple_types]

    @classmethod
    @abstractmethod
    def get_value_types(cls):
        """Return the types of the tuple values"""

    @classmethod
    def is_of_this_type(cls, arg):
        arg_type_ok = isinstance(arg, tuple)
        value_types = cls.get_value_types()
        arg_type_ok = arg_type_ok and len(arg) == len(value_types)
        if arg_type_ok:
            for value, value_type in zip(arg, value_types):
                arg_type_ok = arg_type_ok and value_type.is_of_this_type(value)
                if not arg_type_ok:
                    break
        return arg_type_ok

    @classmethod
    def short_name(cls):
        name = "tuple-" + "-".join(
            tuple_type.short_name() for tuple_type in cls.get_value_types()
        )
        return name

    @classmethod
    def to_scenario_arg(cls, arg):
        scenario_arg_list = []
        value_types = cls.get_value_types()
        for value, value_type in zip(arg, value_types):
            scenario_arg_list.append(value_type.to_scenario_arg(value))
        return tuple(scenario_arg_list)


def TupleType(*value_types):  # pylint: disable=invalid-name
    """TupleType factory method

    Tuples are themselves of type ``tuple`` and they may contain a fixed number of
    elements each one with a fixed type.

    Parameters
    ----------
    value_types : list of `KhiopsTaskArgumentType`
        Type of the tuples value types. The resulting tuple type will admit only tuples
        of the same size of ``value_types``.

    Returns
    -------
    type
        A class which inherits from `AbstractTupleType`.
    """
    # Check the type of the list values
    for value_type in value_types:
        if not issubclass(value_type, KhiopsTaskArgumentType):
            raise TypeError(
                type_error_message(value_type, "value_type", KhiopsTaskArgumentType)
            )
        if not is_simple_type(value_type):
            raise ValueError(
                f"Tuple value type must be simple. It is '{value_type.__name__}'"
            )

    # Return an already registered class or register and return fresh new one
    # pylint: disable=missing-class-docstring
    if AbstractTupleType.is_registered(value_types):
        tuple_type = AbstractTupleType.get_subclass(value_types)
    else:

        class ConcreteTupleType(AbstractTupleType):
            @classmethod
            def get_value_types(cls):
                return value_types

        tuple_type = ConcreteTupleType
        AbstractTupleType.register_subclass(tuple_type)

    return tuple_type


def is_simple_type(arg_type):
    """Returns True if the type is simple

    In this mini type system simple means that they are not generic containers.
    """
    if not issubclass(arg_type, KhiopsTaskArgumentType):
        raise TypeError("arg_type must be a subclass of KhiopsTaskArgumentType")
    return arg_type in [BoolType, IntType, FloatType, StringLikeType]
