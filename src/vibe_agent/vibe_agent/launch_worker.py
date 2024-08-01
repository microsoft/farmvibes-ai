# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import signal
from multiprocessing import set_start_method
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config

from vibe_agent.agent_config import DebugConfig, aks_cosmos_config, local_storage
from vibe_agent.ops import OperationFactoryConfig
from vibe_common.secret_provider import DaprSecretProviderConfig

from .worker import WorkerConfig

local_worker = WorkerConfig(
    factory_spec=OperationFactoryConfig(local_storage, DaprSecretProviderConfig()),
)
aks_worker = WorkerConfig(
    factory_spec=OperationFactoryConfig(aks_cosmos_config, DaprSecretProviderConfig()),
)

LocalWorkerConfig = make_config(impl=local_worker)
AksWorkerConfig = make_config(impl=aks_worker)

WorkerLaunchConfig = make_config(
    "worker",
    debug=DebugConfig(),
    hydra_defaults=["_self_", {"worker": "local"}],
)

cs = ConfigStore.instance()
cs.store(group="worker", name="local", node=LocalWorkerConfig())
cs.store(group="worker", name="aks", node=AksWorkerConfig())
cs.store(name="vibe_worker", node=WorkerLaunchConfig)


@hydra.main(config_path=None, version_base=None, config_name="vibe_worker")
def main(cfg: Any):
    set_start_method("forkserver")
    worker_obj = instantiate(cfg)
    signal.signal(signal.SIGTERM, worker_obj.worker.impl.pre_stop_hook)
    asyncio.run(worker_obj.worker.impl.run())
