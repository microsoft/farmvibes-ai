import os
from typing import Any, Dict

import yaml


def write_yaml(path: str, contents: Dict[str, Any]):
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        yaml.dump(contents, fp)  # type: ignore


def write_file(path: str, contents: str):
    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w") as fp:
        fp.write(contents)
