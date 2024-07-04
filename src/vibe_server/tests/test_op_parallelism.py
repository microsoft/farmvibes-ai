from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from vibe_core.data.core_types import DataVibe, OpIOType
from vibe_server.workflow.runner.runner import OpParallelism
from vibe_server.workflow.workflow import EdgeLabel, EdgeType, GraphNodeType, InputFanOut


class OpSpecMock:
    def __init__(self, inputs: OpIOType):
        self.inputs_spec = {
            name: List[DataVibe] if isinstance(data, list) else DataVibe
            for name, data in inputs.items()
        }


class NodeMock(NamedTuple):
    name: str
    spec: OpSpecMock


@pytest.fixture
def merge_input() -> List[OpIOType]:
    return [{"something": [{"int": i}]} for i in range(10)]


@pytest.fixture
def exploder_input() -> OpIOType:
    return {"to": [{"something": i} for i in range(10)], "other": {"another": "thing"}}


def test_parallelism_merges(merge_input: List[Dict[str, Any]]):
    the_edge = EdgeLabel("from", "to", EdgeType.scatter)
    none = cast(Callable[[GraphNodeType, OpIOType, UUID, int], Awaitable[OpIOType]], None)
    parallelism = OpParallelism([the_edge], cast(GraphNodeType, None), none)
    out = parallelism.fan_in(merge_input)
    assert len(out) == 1
    assert "something" in out
    assert len(out["something"]) == 10


def test_parallelism_explodes_inputs(exploder_input: OpIOType):
    op_mock = cast(GraphNodeType, NodeMock("mock", OpSpecMock(exploder_input)))
    the_edge = EdgeLabel("from", "to", EdgeType.scatter)
    none = cast(Callable[[GraphNodeType, OpIOType, UUID, int], Awaitable[OpIOType]], None)
    parallelism = OpParallelism([the_edge], op_mock, none)
    exploded_inputs = list(parallelism.fan_out(exploder_input))
    assert len(exploded_inputs) == 10


@pytest.mark.anyio
async def test_parallelism_runs(exploder_input: OpIOType):
    async def run_task(_: GraphNodeType, input: OpIOType, __: UUID, ___: int) -> OpIOType:
        return {"out_" + k: v for k, v in input.items()}

    op_mock = cast(GraphNodeType, NodeMock("mock", OpSpecMock(exploder_input)))
    the_edge = EdgeLabel("from", "to", EdgeType.scatter)
    parallelism = OpParallelism([the_edge], op_mock, run_task)
    out = parallelism.fan_in(await parallelism.run(exploder_input, uuid4()))

    assert "out_to" in out
    assert "out_other" in out
    assert len(out["out_to"]) == len(out["out_other"]) == 10


@pytest.mark.anyio
async def test_parallelism_fails(exploder_input: OpIOType):
    async def run_task(_: GraphNodeType, input: OpIOType, __: UUID, ___: int) -> OpIOType:
        raise RuntimeError(":-(")

    op_mock = cast(GraphNodeType, NodeMock("mock", OpSpecMock(exploder_input)))
    the_edge = EdgeLabel("from", "to", EdgeType.scatter)
    parallelism = OpParallelism([the_edge], op_mock, run_task)

    with pytest.raises(RuntimeError):
        await parallelism.run(exploder_input, uuid4())


@patch.object(OpParallelism, "fan_out")
@patch("pydantic.fields.ModelField.validate", side_effect=lambda *args, **_: (args[1], None))
@pytest.mark.anyio
async def test_parallelism_input_fan_out(_: MagicMock, fan_out: MagicMock):
    run_task = MagicMock()
    node = InputFanOut("test", DataVibe)
    parallelism = OpParallelism([], GraphNodeType("test", node), run_task)
    with patch.object(OpParallelism, "fan_in") as fan_in:
        outputs = await parallelism.run(cast(OpIOType, {node.input_port: "👍"}), uuid4())
        fan_in.assert_not_called()
    fan_out.assert_not_called()
    run_task.assert_not_called()
    assert parallelism.fan_in(outputs) == {node.output_port: "👍"}
