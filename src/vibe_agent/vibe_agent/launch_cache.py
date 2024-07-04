import asyncio
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config

from vibe_agent.agent_config import DebugConfig, aks_cosmos_config, local_storage
from vibe_agent.cache import CacheConfig

local_cache = CacheConfig(storage=local_storage, running_on_azure=False)
aks_cache = CacheConfig(storage=aks_cosmos_config, running_on_azure=True)

LocalCacheConfig = make_config(impl=local_cache)
AksCacheConfig = make_config(impl=aks_cache)

CacheLaunchConfig = make_config(
    "cache",
    debug=DebugConfig(),
    hydra_defaults=["_self_", {"cache": "local"}],
)


# Register cache config with hydra's config store
cs = ConfigStore.instance()
cs.store(group="cache", name="local", node=LocalCacheConfig())
cs.store(group="cache", name="aks", node=AksCacheConfig())
cs.store(name="vibe_cache", node=CacheLaunchConfig)


@hydra.main(config_path=None, version_base=None, config_name="vibe_cache")
def main(cfg: Any):
    cache_obj = instantiate(cfg)
    asyncio.run(cache_obj.cache.impl.run())
