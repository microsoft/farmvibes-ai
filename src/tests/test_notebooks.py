import json
import os
from typing import List

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
NOTEBOOK_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "notebooks"))
WIKI_URL = (
    "https://dev.azure.com/ResearchForIndustries/EYWA/_wiki/wikis/EYWA.wiki/214/Notebook-Metadata"
)


def list_notebooks() -> List[str]:
    notebook_list: List[str] = []

    for folder, _, nb_files in os.walk(NOTEBOOK_DIR):
        for nb_file in nb_files:
            if nb_file.endswith(".ipynb"):
                nb_path = os.path.abspath(os.path.join(folder, nb_file))
                notebook_list.append(nb_path)

    return notebook_list


@pytest.mark.parametrize("notebook_path", list_notebooks())
def test_workflows_description(notebook_path: str):
    """Test that all notebooks have name, description and tags metadata"""
    with open(notebook_path) as f:
        nb_json = json.load(f)

    nb_metadata = nb_json["metadata"]
    assert "name" in nb_metadata, f"Missing 'name' metadata, refer to {WIKI_URL}"
    assert "description" in nb_metadata, f"Missing 'description' metadata, refer to {WIKI_URL}"
    assert "disk_space" in nb_metadata, f"Missing disk space requirements, refer to {WIKI_URL}"
    assert "running_time" in nb_metadata, f"Missing expected running time, refer to {WIKI_URL}"
    assert "tags" in nb_metadata, f"Missing tags, refer to {WIKI_URL}"
    assert len(nb_metadata["tags"]) > 0, f"Tag list is empty, refer to {WIKI_URL}"
