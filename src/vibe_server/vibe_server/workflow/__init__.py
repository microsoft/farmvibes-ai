# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from typing import Any, Dict, List, Union

from ..workflow.spec_parser import WorkflowParser, get_workflow_dir
from ..workflow.workflow import Workflow


def get_workflow_path(name: str, base: str = get_workflow_dir()) -> str:
    return os.path.join(base, name) + ".yaml"


def workflow_from_input(input: Union[str, Dict[str, Any]]) -> Workflow:
    workflow: Workflow
    if isinstance(input, str):
        workflow = Workflow.build(get_workflow_path(input))
    else:
        workflow = Workflow(WorkflowParser.parse_dict(input))
    return workflow


def list_workflows() -> List[str]:
    "Returns a list of workflows to be loaded by `load_workflow_by_name`"

    workflow_dir = get_workflow_dir()
    if not os.path.exists(workflow_dir):
        return []

    workflows: List[str] = []
    for dirpath, _, filenames in os.walk(workflow_dir):
        for filename in filenames:
            if filename.endswith(".yaml"):
                workflows.append(
                    re.sub(
                        # Both patterns here are guaranteed to be present
                        # in the input string. We don't want them.
                        f"{workflow_dir}/|\\.yaml",
                        "",
                        os.path.join(dirpath, filename),
                    )
                )
    return workflows
