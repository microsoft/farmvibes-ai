import os
import uuid
from typing import cast

import pytest
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient
from hydra_zen import instantiate

from vibe_agent.storage import (
    BlobAssetManagerConfig,
    CosmosStorage,
    CosmosStorageConfig,
    LocalFileAssetManagerConfig,
    LocalStorageConfig,
)
from vibe_common.secret_provider import KeyVaultSecretConfig

TEST_STORAGE = "https://eywadevtest.blob.core.windows.net"
REMOTE_FILE_CONTAINER = "testdata"
DUMMY_COSMOS_URI = "https://terravibes-db.documents.azure.com:443/"


@pytest.fixture(autouse=True, scope="session")
def stac_container() -> str:
    stac_container_name: str = "stac" + str(uuid.uuid4())[0:6]
    return stac_container_name


@pytest.fixture(autouse=True, scope="session")
def asset_container() -> str:
    asset_name: str = "asset" + str(uuid.uuid4())[0:6]
    return asset_name


@pytest.fixture(scope="session")
def storage_spec(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    stac_container: str,
    asset_container: str,
):
    storage_type: str = request.param  # type: ignore
    if storage_type == "local":
        tmp_path = tmp_path_factory.mktemp("testdir")
        tmp_asset_path = os.path.join(tmp_path, "assets")
        yield LocalStorageConfig(
            local_path=tmp_path, asset_manager=LocalFileAssetManagerConfig(tmp_asset_path)
        )
    elif storage_type == "remote":
        cosmos_asset_container = asset_container + "cosmos"
        key_config = KeyVaultSecretConfig(
            "https://eywa-secrets.vault.azure.net/", "stac-cosmos-write-key", AzureCliCredential()
        )
        key = instantiate(key_config)
        test_db_name = "test-db"
        config = CosmosStorageConfig(
            key=key,
            asset_manager=BlobAssetManagerConfig(
                storage_account_url=TEST_STORAGE,
                storage_account_connection_string="",
                asset_container_name=cosmos_asset_container,
                credential=AzureCliCredential(),
            ),
            cosmos_database_name=test_db_name,
            stac_container_name=stac_container,
            cosmos_url=DUMMY_COSMOS_URI,
        )
        cast(CosmosStorage, instantiate(config)).container_proxy
        yield config
        cred = AzureCliCredential()
        client = BlobServiceClient(TEST_STORAGE, credential=cred)
        asset_client = client.get_container_client(cosmos_asset_container)
        cosmos_client = CosmosClient(config.cosmos_url, key)
        db = cosmos_client.get_database_client(test_db_name)
        db.delete_container(stac_container)
        if asset_client.exists():
            asset_client.delete_container()
    else:
        raise ValueError(f"Invalid storage setup {storage_type}")
