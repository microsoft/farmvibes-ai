# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for working with files."""

import os
from typing import Any, Dict

import yaml


def write_yaml(path: str, contents: Dict[str, Any]):
    """Write contents to a YAML file, creating the parent directory if it doesn't exist yet.

    Args:
        path: The path of the file to write.
        contents: The contents to write to the file.
    """
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        yaml.dump(contents, fp)  # type: ignore


def write_file(path: str, contents: str):
    """Write contents to a file at the given path.

    The function creates the parent directory if it doesn't exist yet.

    Args:
        path: The file path to write to.
        contents: The contents to write in the file.
    """
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        fp.write(contents)
