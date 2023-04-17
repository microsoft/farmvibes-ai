import os
from typing import Any, Dict

import yaml


def write_yaml(path: str, contents: Dict[str, Any]):
    """Writes contents to a YAML file, creating the parent directory if it doesn't exist yet.

    :param path: The path of the file to write.

    :param contents: The contents to write to the file.
    """

    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        yaml.dump(contents, fp)  # type: ignore


def write_file(path: str, contents: str):
    """
    Writes contents to a file at the given path, creating the parent
    directory if it doesn't exist yet.

    :param path: The file path to write to.

    :param contents: The contents to write in the file.
    """
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        fp.write(contents)
