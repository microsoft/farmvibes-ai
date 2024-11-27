# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import codecs
import getpass
import os
import shutil
from datetime import datetime, timezone
from typing import Callable, Union

import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon

from vibe_core.cli.helper import execute_cmd
from vibe_core.cli.local import find_redis_master
from vibe_core.cli.osartifacts import OSArtifacts
from vibe_core.cli.wrappers import KubectlWrapper
from vibe_core.client import FarmvibesAiClient, VibeWorkflowRun, get_default_vibe_client
from vibe_core.datamodel import RunStatus

HOME = os.path.expanduser("~")
DEFAULT_FARMVIBES_CACHE_DATA_DIR = os.path.join(
    os.path.join(HOME, ".cache", "farmvibes-ai"), "data"
)

DELETE_KEY_WITH_PREFIX_CMD = 'redis-cli -a {password} KEYS "{key_prefix}" 2> /dev/null | xargs redis-cli -a {password} DEL 2> /dev/null'  # noqa

RUN_KEY_PREFIX = "run:*"
OP_KEY_PREFIX = "op:*"
ASSET_KEY_PREFIX = "asset:*"


class KubectlRedisWrapper(KubectlWrapper):
    def __init__(self):
        self.cluster_name = os.environ.get(
            "FARMVIBES_AI_CLUSTER_NAME",
            f"farmvibes-ai-{getpass.getuser()}",
        )
        super().__init__(os_artifacts=OSArtifacts(), cluster_name=self.cluster_name)

    def delete_keys_with_prefix(self, prefix: str) -> Union[str, None]:
        result = self.get_secret("redis", ".data.redis-password", self.cluster_name)
        redis_password = codecs.decode(result.encode(), "base64").decode()
        master_pod, redis_master, kind = find_redis_master(self)
        bash_command = DELETE_KEY_WITH_PREFIX_CMD.format(password=redis_password, key_prefix=prefix)
        cmd = [self.os_artifacts.kubectl, "exec", master_pod, "--", "bash", "-c", bash_command]

        retries = 3
        output = None

        for _ in range(retries):
            try:
                output = execute_cmd(cmd, censor_command=True)
                break
            except ValueError:
                continue

        return output


def clear_cache_and_cache_metadata():
    if os.path.exists(DEFAULT_FARMVIBES_CACHE_DATA_DIR):
        shutil.rmtree(DEFAULT_FARMVIBES_CACHE_DATA_DIR)

    redis_via_kubectl = KubectlRedisWrapper()
    redis_via_kubectl.delete_keys_with_prefix(RUN_KEY_PREFIX)
    redis_via_kubectl.delete_keys_with_prefix(OP_KEY_PREFIX)
    redis_via_kubectl.delete_keys_with_prefix(ASSET_KEY_PREFIX)


def ensure_equal_output_images(expected_path: str, actual_path: str):
    with rasterio.open(expected_path) as src:
        expected_ar = (
            src.read()
        )  # Actually read the data. This is a numpy array with shape (bands, height, width)
        expected_profile = src.profile  # Metadata about geolocation, compression, and tiling (dict)
    with rasterio.open(actual_path) as src:
        actual_ar = src.read()
        actual_profile = src.profile
    assert np.allclose(expected_ar, actual_ar)
    assert all(expected_profile[k] == actual_profile[k] for k in expected_profile)


def num_files_in_cache():
    num_files = 0
    for dirpath, dirs, files in os.walk(DEFAULT_FARMVIBES_CACHE_DATA_DIR):
        num_files += len(files)
    return num_files


@pytest.fixture
def helloworld_workflow_fixture():
    clear_cache_and_cache_metadata()

    def run_helloworld_workflow():
        polygon_coords = [
            (-88.062073563448919, 37.081397673802059),
            (-88.026349330507315, 37.085463858128762),
            (-88.026349330507315, 37.085463858128762),
            (-88.012445388773259, 37.069230099135126),
            (-88.035931592028305, 37.048441375086092),
            (-88.068120429075847, 37.058833638440767),
            (-88.062073563448919, 37.081397673802059),
        ]
        polygon = Polygon(polygon_coords)
        start_date = datetime(year=2021, month=2, day=1, tzinfo=timezone.utc)
        end_date = datetime(year=2021, month=2, day=11, tzinfo=timezone.utc)
        client: FarmvibesAiClient = get_default_vibe_client()

        run = client.run(
            "helloworld",
            "test_hello",
            geometry=polygon,
            time_range=(start_date, end_date),
        )

        run.block_until_complete(30)
        return run

    return run_helloworld_workflow


def test_helloworld_once(helloworld_workflow_fixture: Callable[[], VibeWorkflowRun]):
    run = helloworld_workflow_fixture()

    assert run.status == RunStatus.done, f"Workflow did not finish successfully. {run.task_details}"
    assert run.output is not None, "Workflow did not produce output"

    ensure_equal_output_images(
        os.path.join(os.path.dirname(__file__), "expected.tif"),
        run.output["raster"][0].assets[0].local_path,  # type: ignore
    )


def test_helloworld_workflow_twice(helloworld_workflow_fixture: Callable[[], VibeWorkflowRun]):
    # when run twice result should be cached and output should be the same file

    run1 = helloworld_workflow_fixture()
    assert (
        run1.status == RunStatus.done
    ), f"Workflow did not finish successfully. {run1.task_details}"
    assert run1.output is not None, "Workflow did not produce output"
    run1_raster_path = run1.output["raster"][0].assets[0].local_path  # type: ignore

    run2 = helloworld_workflow_fixture()
    assert (
        run2.status == RunStatus.done
    ), f"Workflow did not finish successfully. {run2.task_details}"
    assert run2.output is not None, "Workflow did not produce output"
    run2_raster_path = run2.output["raster"][0].assets[0].local_path  # type: ignore

    assert run1_raster_path == run2_raster_path


def test_run_helloworld_once_delete(helloworld_workflow_fixture: Callable[[], VibeWorkflowRun]):
    run = helloworld_workflow_fixture()
    assert run.status == RunStatus.done, f"Workflow did not finish successfully. {run.task_details}"
    assert run.output is not None, "Workflow did not produce output"
    assert os.path.exists(run.output["raster"][0].assets[0].local_path)  # type: ignore

    run.delete()
    run.block_until_deleted(20)
    assert (
        run.status == RunStatus.deleted
    ), f"Workflow was not deleted successfully. {run.task_details}"
    assert 0 == num_files_in_cache()


def test_run_helloworld_twice_delete(helloworld_workflow_fixture: Callable[[], VibeWorkflowRun]):
    run1 = helloworld_workflow_fixture()
    assert (
        run1.status == RunStatus.done
    ), f"Workflow did not finish successfully. {run1.task_details}"
    assert run1.output is not None, "Workflow did not produce output"

    run2 = helloworld_workflow_fixture()
    assert (
        run2.status == RunStatus.done
    ), f"Workflow did not finish successfully. {run2.task_details}"

    num_files_in_cache_before_delete = num_files_in_cache()

    run1.delete()
    run1.block_until_deleted(20)

    assert (
        run1.status == RunStatus.deleted
    ), f"Workflow was not deleted successfully. {run1.task_details}"

    assert num_files_in_cache_before_delete == num_files_in_cache()
    assert os.path.exists(run2.output["raster"][0].assets[0].local_path)  # type: ignore
