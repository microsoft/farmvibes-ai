import asyncio
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config

from vibe_agent.agent_config import DebugConfig, aks_cosmos_config, local_storage
from vibe_agent.cache_metadata_store import RedisCacheMetadataStoreConfig
from vibe_agent.data_ops import DataOpsConfig

# Create instiatiatable configs for CacheMetadataStoreProtocol
redis_cache_metadata_store_config = RedisCacheMetadataStoreConfig()

# create two DataOpsConfigs: one to build DataOpsManager with local storage and another for
# to build DataOpsManager with AKS/Cosmos storage
local_data_ops_config = DataOpsConfig(
    metadata_store=redis_cache_metadata_store_config, storage=local_storage
)
aks_data_ops_config = DataOpsConfig(
    metadata_store=redis_cache_metadata_store_config, storage=aks_cosmos_config
)

# two configs each with one field, impl, one set to the DataOpsConfig for local storage, the
# other for AKS/Cosmos
LocalDataOpsConfig = make_config(impl=local_data_ops_config)
AksDataOpsConfig = make_config(impl=aks_data_ops_config)

# launching the data ops service has two parts that need to be configured:
# 1. whether or not we are debugging the service
# 2. should the DataOpsManager be referencing local storage or a AKS/Cosmos storage
#    - by default, it will use the "local" entry in the "data_ops" group in the ConfigStore as the
#      default config for the data_ops field
DataOpsLaunchConfig = make_config(
    "data_ops",
    debug=DebugConfig(),
    hydra_defaults=["_self_", {"data_ops": "local"}],
)

# Register configs config with hydra's config store
cs = ConfigStore.instance()
cs.store(group="data_ops", name="local", node=LocalDataOpsConfig)
cs.store(group="data_ops", name="aks", node=AksDataOpsConfig)
cs.store(name="vibe_data_ops", node=DataOpsLaunchConfig)


# The @hydra_main decorator in Hydra resolves all missing configurations from the top-level
# configuration using entries in the config store. If a configuration value is missing, Hydra
# will search the config store for a matching key and use the value stored in the config store
# if one is found.
@hydra.main(config_path=None, version_base=None, config_name="vibe_data_ops")
def main(cfg: Any):
    data_ops_launch_config_obj = instantiate(cfg)
    asyncio.run(data_ops_launch_config_obj.data_ops.impl.run())
