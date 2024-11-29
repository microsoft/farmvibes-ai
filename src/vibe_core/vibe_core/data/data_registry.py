"""Data registry types and functions used in FarmVibes.AI."""

import warnings
from typing import Any, Dict, Type, TypeVar, cast

GenericTypeVibe = TypeVar("GenericTypeVibe", bound=Type[Any])
"""Generic type for data registry."""

__DATA_REGISTRY: Dict[str, Type[Any]] = {}


def register_vibe_datatype(classdef: GenericTypeVibe) -> GenericTypeVibe:
    """Register a class as a data type in the FarmVibes.AI data registry.

    Args:
        classdef: The class to register.

    Returns:
        The class.
    """
    id = get_name(classdef)
    if id in __DATA_REGISTRY:
        warnings.warn(f"Class {id} already registered.", DeprecationWarning, stacklevel=2)
        return cast(GenericTypeVibe, retrieve(id))
    __DATA_REGISTRY[id] = classdef

    return cast(GenericTypeVibe, classdef)


def retrieve(id: str) -> Type[Any]:
    """Retrieve a registered data type from the FarmVibes.AI data registry.

    Args:
        id: The ID of the data type to retrieve.

    Returns:
        The registered data type.
    """
    return __DATA_REGISTRY[id]


def get_name(classdef: Type[Any]) -> str:
    """Get the name of a class.

    Args:
        classdef: The class to get the name of.

    Returns:
        The name of the class.
    """
    return classdef.__name__


def get_id(classdef: Type[Any]) -> str:
    """Get the ID of a class.

    Args:
        classdef: The class to get the ID of.

    Returns:
        The ID of the class.
    """
    id = get_name(classdef)
    return id
