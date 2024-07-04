import logging
import os

import debugpy
from hydra_zen import builds

from vibe_agent.storage.asset_management import BlobAssetManagerConfig
from vibe_agent.storage.local_storage import (
    LocalFileAssetManagerConfig,
    LocalStorageConfig,
)
from vibe_agent.storage.remote_storage import CosmosStorageConfig
from vibe_common.constants import (
    DEFAULT_ASSET_PATH,
    DEFAULT_CATALOG_PATH,
    DEFAULT_SECRET_STORE_NAME,
)
from vibe_common.secret_provider import DaprSecretConfig


def setup_debug(activate: bool = False, port: int = 5678):
    if not activate:
        return

    debugpy.listen(port)
    logging.info(f"Debugger enabled and listening on port {port}")


DebugConfig = builds(setup_debug, populate_full_signature=True)

local_storage = LocalStorageConfig(
    local_path=DEFAULT_CATALOG_PATH,
    asset_manager=LocalFileAssetManagerConfig(DEFAULT_ASSET_PATH),
)

stac_cosmos_uri = DaprSecretConfig(
    store_name=DEFAULT_SECRET_STORE_NAME,
    secret_name=os.environ["STAC_COSMOS_URI_SECRET"],
    key_name=os.environ["STAC_COSMOS_URI_SECRET"],
)

stac_cosmos_key = DaprSecretConfig(
    store_name=DEFAULT_SECRET_STORE_NAME,
    secret_name=os.environ["STAC_COSMOS_CONNECTION_KEY_SECRET"],
    key_name=os.environ["STAC_COSMOS_CONNECTION_KEY_SECRET"],
)

stac_cosmos_db = DaprSecretConfig(
    store_name=DEFAULT_SECRET_STORE_NAME,
    secret_name=os.environ["STAC_COSMOS_DATABASE_NAME_SECRET"],
    key_name=os.environ["STAC_COSMOS_DATABASE_NAME_SECRET"],
)

stac_cosmos_container = DaprSecretConfig(
    store_name=DEFAULT_SECRET_STORE_NAME,
    secret_name=os.environ["STAC_CONTAINER_NAME_SECRET"],
    key_name=os.environ["STAC_CONTAINER_NAME_SECRET"],
)

try:
    storage_account_url = DaprSecretConfig(
        store_name=DEFAULT_SECRET_STORE_NAME,
        secret_name=os.environ["BLOB_STORAGE_ACCOUNT_URL"],
        key_name=os.environ["BLOB_STORAGE_ACCOUNT_URL"],
    )
except Exception:
    storage_account_url = ""

try:
    storage_account_connection_string = DaprSecretConfig(
        store_name=DEFAULT_SECRET_STORE_NAME,
        secret_name=os.environ["BLOB_STORAGE_ACCOUNT_CONNECTION_STRING"],
        key_name=os.environ["BLOB_STORAGE_ACCOUNT_CONNECTION_STRING"],
    )
except Exception:
    storage_account_connection_string = ""


aks_asset_manager = BlobAssetManagerConfig(
    storage_account_url=storage_account_url,
    storage_account_connection_string=storage_account_connection_string,
    asset_container_name=os.environ["BLOB_CONTAINER_NAME"],
    credential=None,
    max_upload_concurrency=6,
)

aks_cosmos_config = CosmosStorageConfig(
    key=stac_cosmos_key,
    asset_manager=aks_asset_manager,
    stac_container_name=stac_cosmos_container,
    cosmos_database_name=stac_cosmos_db,
    cosmos_url=stac_cosmos_uri,
)
