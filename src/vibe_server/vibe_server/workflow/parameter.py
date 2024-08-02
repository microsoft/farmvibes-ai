# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .spec_parser import (
    OperationSpec,
    TaskType,
    WorkflowSpec,
    WorkflowSpecNode,
    get_parameter_reference,
)


class ParameterResolver:
    def __init__(self, workflows_dir: str, ops_dir: str):
        self.workflows_dir = workflows_dir
        self.ops_dir = ops_dir

    def resolve(self, workflow_spec: WorkflowSpec) -> Dict[str, "Parameter"]:
        wf_params = self._get_wf_params(workflow_spec, "root")
        return {p.name: p for p in wf_params}

    def _get_wf_params(self, workflow_spec: WorkflowSpec, task_name: str):
        wf_params: List[Parameter] = []
        for k, v in workflow_spec.parameters.items():
            default = workflow_spec.default_parameters[k]
            descriptions = workflow_spec.description.parameters
            desc = descriptions.get(k) if descriptions is not None else None
            wf_params.append(
                Parameter(name=k, task=task_name, value=v, default=default, description=desc)
            )
        # Get references from tasks
        refs: Dict[str, List[Parameter]] = defaultdict(list)
        for task_name, node in workflow_spec.tasks.items():
            for task_param in self._get_node_params(node):
                ref = task_param.reference
                if ref is not None:
                    refs[ref].append(task_param)
        for wf_param in wf_params:
            for ref_param in refs[wf_param.name]:
                wf_param.add_child(ref_param)
        return wf_params

    def _get_op_params(self, op_spec: OperationSpec, task_name: str) -> List["Parameter"]:
        def foo(
            params: Dict[str, Any],
            descriptions: Optional[Dict[str, Any]],
            defaults: Dict[str, Any],
            prefix: str = "",
        ):
            for k, v in params.items():
                desc = descriptions.get(k) if descriptions is not None else None
                default = defaults[k]
                if isinstance(v, dict):
                    assert isinstance(desc, dict) or desc is None
                    assert isinstance(default, dict)
                    for p in foo(v, desc, default, prefix=k):
                        yield p
                else:
                    assert isinstance(desc, str) or desc is None
                    name = f"{prefix}.{k}" if prefix else k
                    yield Parameter(
                        name=name, task=task_name, value=v, default=default, description=desc
                    )

        return [
            p
            for p in foo(
                op_spec.parameters, op_spec.description.parameters, op_spec.default_parameters
            )
        ]

    def _get_node_params(self, node: WorkflowSpecNode):
        task = node.load(ops_base_dir=self.ops_dir, workflow_dir=self.workflows_dir)
        if node.type == TaskType.op:
            return self._get_op_params(cast(OperationSpec, task), node.task)
        return self._get_wf_params(cast(WorkflowSpec, task), node.task)


class Parameter:
    def __init__(
        self,
        name: str,
        task: str,
        value: Any,
        default: Any,
        description: Optional[Union[str, Dict[str, str]]],
    ) -> None:
        self.name = name
        self.task = task
        self._value = value
        self._default = default
        self._description = description
        self.childs: List["Parameter"] = []

    def add_child(self, child: "Parameter"):
        self.childs.append(child)

    def _resolve(self, attr: str, private_attr: str):
        # If our attribute is None and we have childs, lets get the default value from them
        if getattr(self, private_attr) is None and self.childs:
            attrs = []
            for p in self.childs:
                p_attr = getattr(p, attr)
                if not isinstance(p_attr, tuple):
                    p_attr = (p_attr,)
                for i in p_attr:
                    if i not in attrs:
                        attrs.append(i)
            if len(attrs) == 1:
                return attrs[0]
            return tuple(attrs)
        return getattr(self, private_attr)

    @property
    def default(self) -> Any:
        return self._resolve("default", "_default")

    @property
    def description(self) -> Union[str, Tuple[str], None]:
        descriptions = self._resolve("description", "_description")
        # Discard `None` from children and adjust accordingly
        if isinstance(descriptions, tuple):
            descriptions = tuple(d for d in descriptions if d is not None)
            if not descriptions:  # Empty set, return None
                return None
            if len(descriptions) == 1:
                return descriptions[0]
        return descriptions

    @property
    def reference(self) -> Optional[str]:
        return get_parameter_reference(self._value, self.task)
