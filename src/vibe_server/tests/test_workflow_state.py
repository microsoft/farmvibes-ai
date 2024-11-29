# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import pytest

from vibe_core.datamodel import RunDetails, RunStatus
from vibe_server.orchestrator import WorkflowStateUpdate
from vibe_server.workflow.runner import WorkflowChange

MOCK_SUBMISSION_TIME = datetime(2020, 1, 2, 3, 4, 5, 6)


async def setup_updater(run_config: Dict[str, Any], tasks: List[str]):
    with patch.object(WorkflowStateUpdate, "_init_cache", autospec=True) as mock_method:
        deets = asdict(RunDetails())
        deets["submission_time"] = MOCK_SUBMISSION_TIME

        def mock_fun(self):  # type:ignore
            self.wf_cache["details"] = deets
            self._cache_init = True

        mock_method.side_effect = mock_fun
        updater = WorkflowStateUpdate(run_config["id"])
        await updater(WorkflowChange.WORKFLOW_STARTED, tasks=tasks)
    return updater


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_workflow_started(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    transaction_ops = transaction.mock_calls[0][1][0]
    # We update all tasks + workflow
    assert len(transaction_ops) == len(tasks) + 1
    assert transaction_ops[-1]["key"] == str(updater.run_id)
    wf_cache = updater._get_cache(None, None)[0]
    assert wf_cache["status"] == RunStatus.pending
    assert wf_cache["submission_time"] == MOCK_SUBMISSION_TIME
    assert wf_cache["start_time"] is not None
    for t_op, task in zip(transaction_ops, tasks):
        assert task in updater.task_cache
        cache = updater._get_cache(task, None)[0]
        assert t_op["key"] == f"{updater.run_id}-{task}"
        assert cache["status"] == RunStatus.pending
        assert cache["subtasks"] is None
        assert cache["submission_time"] is None


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_workflow_finished(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    await updater(WorkflowChange.WORKFLOW_FINISHED)
    transaction_ops = transaction.mock_calls[1][1][0]
    # We only update the workflow
    assert len(transaction_ops) == 1
    assert transaction_ops[0]["key"] == str(updater.run_id)


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_task_started(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    task_start = "task1"
    num_subtasks = 4
    await updater(WorkflowChange.TASK_STARTED, task=task_start, num_subtasks=num_subtasks)
    transaction_ops = transaction.mock_calls[1][1][0]
    # We update the task, not the workflow (still pending)
    assert len(transaction_ops) == 1
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{task_start}"
    cache = updater._get_cache(task_start, None)[0]
    assert cache["status"] == RunStatus.pending
    assert len(cache["subtasks"]) == num_subtasks
    assert all([s["status"] == RunStatus.pending for s in cache["subtasks"]])


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_propagate_up(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    assert updater._get_cache(None, None)[0]["submission_time"] == MOCK_SUBMISSION_TIME
    assert updater._get_cache(None, None)[0]["start_time"] is not None
    task_start = "task1"
    num_subtasks = 4
    await updater(WorkflowChange.TASK_STARTED, task=task_start, num_subtasks=num_subtasks)
    transaction.reset_mock()
    await updater(WorkflowChange.SUBTASK_QUEUED, task=task_start, subtask_idx=0)
    transaction_ops = transaction.mock_calls[0][1][0]
    # We update the task and workflow to queued
    assert len(transaction_ops) == 2
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{task_start}"
    assert transaction_ops[1]["key"] == f"{updater.run_id}"

    assert updater._get_cache(None, None)[0]["status"] == RunStatus.queued
    assert updater._get_cache(task_start, None)[0]["status"] == RunStatus.queued
    assert updater._get_cache(task_start, 0)[0]["status"] == RunStatus.queued
    # Check that submission time was properly updated
    subtask_submission = updater._get_cache(task_start, 0)[0]["submission_time"]
    assert subtask_submission is not None
    assert updater._get_cache(task_start, None)[0]["submission_time"] == subtask_submission

    # A different subtask should still be pending
    assert updater._get_cache(task_start, 1)[0]["status"] == RunStatus.pending

    # Let's queue another subtask from the same task
    await updater(WorkflowChange.SUBTASK_QUEUED, task=task_start, subtask_idx=1)
    transaction_ops = transaction.mock_calls[-1][1][0]
    # We only update the task since the workflow is already queued
    assert len(transaction_ops) == 1
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{task_start}"
    assert updater._get_cache(task_start, 1)[0]["status"] == RunStatus.queued

    # Let's start the other task and queue a subtask
    other_task = "task2"
    await updater(WorkflowChange.TASK_STARTED, task=other_task, num_subtasks=1)
    transaction_ops = transaction.mock_calls[-1][1][0]
    assert len(transaction_ops) == 1
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{other_task}"
    await updater(WorkflowChange.SUBTASK_QUEUED, task=other_task, subtask_idx=0)
    transaction_ops = transaction.mock_calls[-1][1][0]
    assert len(transaction_ops) == 1
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{other_task}"

    # Let's start a subtask on the first task
    await updater(WorkflowChange.SUBTASK_RUNNING, task=task_start, subtask_idx=0)
    transaction_ops = transaction.mock_calls[-1][1][0]
    # We should update the task and the workflow to running here
    assert len(transaction_ops) == 2
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{task_start}"
    assert transaction_ops[1]["key"] == f"{updater.run_id}"
    assert updater._get_cache(task_start, 0)[0]["status"] == RunStatus.running
    assert updater._get_cache(task_start, None)[0]["status"] == RunStatus.running
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.running
    # The start times should match
    subtask_start = updater._get_cache(task_start, 0)[0]["start_time"]
    assert updater._get_cache(task_start, None)[0]["start_time"] == subtask_start

    # Let's finish the first subtask
    await updater(WorkflowChange.SUBTASK_FINISHED, task=task_start, subtask_idx=0)
    transaction_ops = transaction.mock_calls[-1][1][0]
    # We should update the task and the workflow back to queued
    assert len(transaction_ops) == 2
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{task_start}"
    assert transaction_ops[1]["key"] == f"{updater.run_id}"
    assert updater._get_cache(task_start, 0)[0]["status"] == RunStatus.done
    assert updater._get_cache(task_start, None)[0]["status"] == RunStatus.queued
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.queued
    # The task should have an end time, but the task and workflow should not be updated
    assert updater._get_cache(task_start, 0)[0]["end_time"] is not None
    assert updater._get_cache(task_start, None)[0]["end_time"] is None
    assert updater._get_cache(None, None)[0]["end_time"] is None

    # If we start the subtask for the other task, the workflow should be running
    await updater(WorkflowChange.SUBTASK_RUNNING, task=other_task, subtask_idx=0)
    transaction_ops = transaction.mock_calls[-1][1][0]
    assert len(transaction_ops) == 2
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{other_task}"
    assert transaction_ops[1]["key"] == f"{updater.run_id}"
    assert updater._get_cache(other_task, 0)[0]["status"] == RunStatus.running
    assert updater._get_cache(other_task, None)[0]["status"] == RunStatus.running
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.running

    # Completing the only subtask should set the task to finished and workflow back to queued
    await updater(WorkflowChange.SUBTASK_FINISHED, task=other_task, subtask_idx=0)
    transaction_ops = transaction.mock_calls[-1][1][0]
    assert len(transaction_ops) == 2
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{other_task}"
    assert transaction_ops[1]["key"] == f"{updater.run_id}"
    assert updater._get_cache(other_task, 0)[0]["status"] == RunStatus.done
    assert updater._get_cache(other_task, None)[0]["status"] == RunStatus.done
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.queued
    # The task should have an end time, but the workflow should not be updated
    subtask_end = updater._get_cache(other_task, 0)[0]["end_time"]
    assert subtask_end is not None
    assert updater._get_cache(other_task, None)[0]["end_time"] == subtask_end
    assert updater._get_cache(None, None)[0]["end_time"] is None

    # Complete all subtasks for the first task
    for subtask_idx in range(num_subtasks):
        await updater(WorkflowChange.SUBTASK_FINISHED, task=task_start, subtask_idx=subtask_idx)
        assert updater._get_cache(task_start, subtask_idx)[0]["status"] == RunStatus.done
    # The task should be finished and the workflow should NOT
    assert updater._get_cache(task_start, None)[0]["status"] == RunStatus.done
    assert updater._get_cache(None, None)[0]["status"] != RunStatus.done
    # Check end time for the task
    subtask_end = updater._get_cache(task_start, 3)[0]["end_time"]
    assert subtask_end is not None
    assert updater._get_cache(task_start, None)[0]["end_time"] == subtask_end
    assert updater._get_cache(None, None)[0]["end_time"] is None


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_workflow_cancel(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    task_start = tasks[0]
    num_subtasks = 4
    finished_subtask = 2
    await updater(WorkflowChange.TASK_STARTED, task=task_start, num_subtasks=num_subtasks)
    await updater(WorkflowChange.SUBTASK_FINISHED, task=task_start, subtask_idx=finished_subtask)
    transaction.reset_mock()
    await updater(WorkflowChange.WORKFLOW_CANCELLED)
    transaction_ops = transaction.mock_calls[0][1][0]
    # We update the workflow and all tasks
    assert len(transaction_ops) == 3
    for t_op, task in zip(transaction_ops, tasks):
        assert t_op["key"] == f"{updater.run_id}-{task}"
    assert transaction_ops[-1]["key"] == str(updater.run_id)
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.cancelled
    assert updater._get_cache(None, None)[0]["reason"] == updater.user_request_reason
    for task, task_cache in updater.task_cache.items():
        assert task in tasks
        assert task_cache["status"] == RunStatus.cancelled
        assert task_cache["reason"] == updater.user_request_reason
    subtasks = updater._get_cache(task_start, None)[0]["subtasks"]
    # We should have cancelled all subtasks except the one that finished
    for i, subtask in enumerate(subtasks):
        if i == finished_subtask:
            assert subtask["status"] == RunStatus.done
        else:
            assert subtask["status"] == RunStatus.cancelled


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_no_update_if_done(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    task_start = tasks[0]
    num_subtasks = 4
    canceled_subtask = 0
    finished_subtask = 2
    await updater(WorkflowChange.TASK_STARTED, task=task_start, num_subtasks=num_subtasks)
    await updater(WorkflowChange.SUBTASK_FINISHED, task=task_start, subtask_idx=finished_subtask)
    await updater(WorkflowChange.WORKFLOW_CANCELLED)
    transaction.reset_mock()

    # We should not update anything if we try to update a finished task
    # Either if it's marked as `done`
    await updater(WorkflowChange.SUBTASK_RUNNING, task=task_start, subtask_idx=finished_subtask)
    transaction.assert_not_called()
    # Or if it's marked as `cancelled`
    await updater(WorkflowChange.SUBTASK_RUNNING, task=task_start, subtask_idx=canceled_subtask)
    transaction.assert_not_called()


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_workflow_fail(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2"]
    updater = await setup_updater(run_config, tasks)
    task_start = tasks[0]
    num_subtasks = 1
    finished_subtask = 0
    await updater(WorkflowChange.WORKFLOW_STARTED, tasks=tasks)
    await updater(WorkflowChange.TASK_STARTED, task=task_start, num_subtasks=num_subtasks)
    await updater(WorkflowChange.SUBTASK_FINISHED, task=task_start, subtask_idx=finished_subtask)
    transaction.reset_mock()
    failure_reason = "Something went wrong 💀"
    await updater(WorkflowChange.WORKFLOW_FAILED, reason=failure_reason)
    transaction_ops = transaction.mock_calls[0][1][0]
    # We update the workflow and one task
    assert len(transaction_ops) == 2
    assert transaction_ops[0]["key"] == f"{updater.run_id}-{tasks[1]}"
    assert transaction_ops[-1]["key"] == str(updater.run_id)
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.failed
    # We should have the reason of failure here
    assert updater._get_cache(None, None)[0]["reason"] == failure_reason
    # The first task should be done
    assert updater._get_cache(task_start, None)[0]["status"] == RunStatus.done
    assert updater._get_cache(task_start, 0)[0]["status"] == RunStatus.done
    # The second task should be cancelled
    assert updater._get_cache(tasks[1], None)[0]["status"] == RunStatus.cancelled
    # We should have the cancellation reason for workflow failure here
    assert updater._get_cache(tasks[1], None)[0]["reason"] == updater.workflow_failure_reason


@patch("vibe_common.statestore.StateStore.transaction")
@patch("vibe_common.statestore.StateStore.retrieve")
@patch("vibe_common.statestore.StateStore.store")
@pytest.mark.anyio
async def test_subtask_fail(
    store: Mock, retrieve: Mock, transaction: Mock, run_config: Dict[str, Any]
):
    retrieve.return_value = run_config
    tasks = ["task1", "task2", "task3", "task4"]
    updater = await setup_updater(run_config, tasks)
    # Task with several subtasks
    await updater(WorkflowChange.TASK_STARTED, task=tasks[0], num_subtasks=3)
    # Task with a single subtask
    await updater(WorkflowChange.TASK_STARTED, task=tasks[1], num_subtasks=1)
    # Task with no subtasks
    # Task with single subtask that's done
    await updater(WorkflowChange.TASK_STARTED, task=tasks[3], num_subtasks=1)
    await updater(WorkflowChange.SUBTASK_FINISHED, task=tasks[3], subtask_idx=0)
    # First task has a subtask that is done, and one that is not
    await updater(WorkflowChange.SUBTASK_FINISHED, task=tasks[0], subtask_idx=0)
    await updater(WorkflowChange.SUBTASK_QUEUED, task=tasks[0], subtask_idx=1)
    transaction.reset_mock()
    # The last subtask fails
    failure_reason = "Something went wrong 💀"
    await updater(
        WorkflowChange.SUBTASK_FAILED, task=tasks[0], subtask_idx=2, reason=failure_reason
    )
    transaction_ops = transaction.mock_calls[0][1][0]
    expected_cancel_reason = f"Cancelled because task '{tasks[0]}' (subtask 2) failed"
    # We update the workflow and three tasks
    assert len(transaction_ops) == 4
    for t_op, task in zip(transaction_ops, tasks[:-1]):
        assert t_op["key"] == f"{updater.run_id}-{task}"
    assert transaction_ops[-1]["key"] == str(updater.run_id)
    # Workflow is marked as failed
    assert updater._get_cache(None, None)[0]["status"] == RunStatus.failed
    # We should have the reason of failure here
    assert updater._get_cache(None, None)[0]["reason"] == failure_reason
    # The first task should be failed
    assert updater._get_cache(tasks[0], None)[0]["status"] == RunStatus.failed
    assert updater._get_cache(tasks[0], None)[0]["reason"] == failure_reason
    # Last subtask should be failed
    assert updater._get_cache(tasks[0], 2)[0]["status"] == RunStatus.failed
    assert updater._get_cache(tasks[0], 2)[0]["reason"] == failure_reason
    # The first subtask should be done still
    assert updater._get_cache(tasks[0], 0)[0]["status"] == RunStatus.done
    # The second subtask should be cancelled
    assert updater._get_cache(tasks[0], 1)[0]["status"] == RunStatus.cancelled
    assert updater._get_cache(tasks[0], 1)[0]["reason"] == expected_cancel_reason
    # Other unfinished tasks should be cancelled
    for task in tasks[1:-1]:
        assert updater._get_cache(task, None)[0]["status"] == RunStatus.cancelled
    # Last task should be done
    assert updater._get_cache(tasks[-1], None)[0]["status"] == RunStatus.done


@patch.object(WorkflowStateUpdate, "commit_cache_for")
@pytest.mark.anyio
async def test_workflow_state_update_subtasks(commit: Mock, run_config: Dict[str, Any]):
    op_name = "fake-op"
    updater = await setup_updater(run_config, [op_name])
    await updater(WorkflowChange.TASK_STARTED, task=op_name, num_subtasks=3)
    subtasks = updater.task_cache[op_name]["subtasks"]
    assert len(subtasks) == 3
    assert all(r["status"] == RunStatus.pending for r in subtasks)
    RunDetails(**subtasks[0])

    def compare(values: Tuple[int, int, int, int]):
        counts = Counter([r["status"] for r in subtasks])
        return all(
            counts[k] == v
            for k, v in zip(
                (RunStatus.pending, RunStatus.queued, RunStatus.running, RunStatus.done), values
            )
        )

    await updater(WorkflowChange.SUBTASK_QUEUED, task=op_name, subtask_idx=0)
    assert subtasks[0]["status"] == RunStatus.queued
    RunDetails(**subtasks[0])
    compare((2, 1, 0, 0))
    await updater(WorkflowChange.SUBTASK_QUEUED, task=op_name, subtask_idx=2)
    assert subtasks[2]["status"] == RunStatus.queued
    compare((1, 2, 0, 0))
    await updater(WorkflowChange.SUBTASK_RUNNING, task=op_name, subtask_idx=0)
    assert subtasks[0]["status"] == RunStatus.running
    RunDetails(**subtasks[0])
    compare((1, 1, 1, 0))
    await updater(WorkflowChange.SUBTASK_RUNNING, task=op_name, subtask_idx=2)
    assert subtasks[2]["status"] == RunStatus.running
    compare((1, 0, 2, 0))
    await updater(WorkflowChange.SUBTASK_FINISHED, task=op_name, subtask_idx=2)
    assert subtasks[2]["status"] == RunStatus.done
    RunDetails(**subtasks[2])
    compare((1, 0, 1, 1))
    await updater(WorkflowChange.SUBTASK_QUEUED, task=op_name, subtask_idx=1)
    assert subtasks[1]["status"] == RunStatus.queued
    compare((0, 1, 1, 1))
