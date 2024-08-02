# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import copy
from typing import Dict, List

from vibe_core.data.core_types import InnerIOType, OpIOType

from ..workflow import GraphNodeType, Workflow


class TaskIOHandler:
    IoMapType = Dict[GraphNodeType, Dict[str, List[InnerIOType]]]
    input_map: IoMapType
    output_map: IoMapType
    source_map: IoMapType
    sink_map: IoMapType

    @staticmethod
    def _update_dict(task: GraphNodeType, input_name: str, d: IoMapType, value: List[InnerIOType]):
        if task in d:
            d[task][input_name] = value
        else:
            d[task] = {input_name: value}

    def _attach_input_port(self, node: GraphNodeType, input_port: str, io: List[InnerIOType]):
        self.input_map.setdefault(node, {})
        node_inputs = self.input_map[node]
        if input_port in node_inputs:
            raise ValueError(
                f"Tried to attach input port {node.name}.{input_port} but it is already attached"
            )
        node_inputs[input_port] = io

    def _parse_workflow(self, workflow: Workflow):
        io: List[InnerIOType]
        for origin, destination, label in workflow.edges:
            if origin in self.output_map and label.srcport in self.output_map[origin]:
                io = self.output_map[origin][label.srcport]
            else:
                io = []
                self._update_dict(origin, label.srcport, self.output_map, io)
            self._attach_input_port(destination, label.dstport, io)

        for sink, ports in workflow.sinks.items():
            for port in ports:
                if sink in self.output_map and port in self.output_map[sink]:
                    # sink already exists as input to another task
                    io = self.output_map[sink][port]
                else:
                    # new output that is a sink only
                    io = []
                    self._update_dict(sink, port, self.output_map, io)
                self._update_dict(sink, port, self.sink_map, io)

        for source, ports in workflow.sources.items():
            for port in ports:
                io = []
                self._attach_input_port(source, port, io)
                self._update_dict(source, port, self.source_map, io)

    def __init__(self, workflow: Workflow):
        self.input_map = {}
        self.output_map = {}
        self.sink_map = {}
        self.source_map = {}
        self._parse_workflow(workflow)

    def add_result(self, task: GraphNodeType, value: OpIOType):
        for output_name, result in value.items():
            # Calling `get` here may create a new dict/list but, if it is new,
            # it won't be consumed by any other task, or sink
            io = self.output_map.get(task, {}).get(output_name, [])
            if len(io) != 0:
                raise RuntimeError(f"Repeated write to task '{task}' output '{output_name}'.")
            io.append(result)

    def retrieve_input(self, task: GraphNodeType) -> OpIOType:
        input_dict: OpIOType = {}
        for kw_name, input_value in self.input_map[task].items():
            input_dict[kw_name] = copy(input_value[0])

        return input_dict

    def add_sources(self, values: OpIOType):
        if len(values) != sum([len(t) for t in self.source_map.values()]):
            raise ValueError("Tried to add different number of values to workflow")

        for task, ports in self.source_map.items():
            for port in ports:
                key = task.name + "." + port
                try:
                    value = values.pop(key)
                    ports[port].append(value)
                except KeyError:
                    raise ValueError(f"Unable to find source {key} for running workflow")

        if values:
            raise ValueError(f"Tried to add unknown values {values.keys()} to workflow")

    def retrieve_sinks(self) -> OpIOType:
        output_dict: OpIOType = {}
        for task, sink_outputs in self.sink_map.items():
            for task_output_name, sink_output in sink_outputs.items():
                output_dict[task.name + "." + task_output_name] = copy(sink_output[0])

        return output_dict

    def __del__(self):
        for mapping in (self.input_map, self.output_map, self.sink_map, self.source_map):
            for ports in mapping.values():
                for port in ports:
                    try:
                        ports[port].pop()
                    except IndexError:
                        break
        del self.input_map
        del self.output_map
        del self.sink_map
        del self.source_map


class WorkflowIOHandler:
    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    def map_input(self, input_items: OpIOType) -> OpIOType:
        return {
            node: input_items[key]
            for key, nodes in self.workflow.source_mappings.items()
            for node in nodes
        }

    def map_output(self, output_items: OpIOType) -> OpIOType:
        return {key: output_items[value] for key, value in self.workflow.sink_mappings.items()}
