import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, BlobProperties, BlobServiceClient, ContainerClient
from hydra_zen import MISSING, builds

from vibe_common.constants import DEFAULT_BLOB_ASSET_MANAGER_CONTAINER
from vibe_common.tokens import BlobTokenManagerConnectionString, BlobTokenManagerCredentialed
from vibe_core.file_downloader import download_file
from vibe_core.uri import is_local, local_uri_to_path, uri_to_filename

from .file_upload import local_upload, remote_upload

CACHE_SIZE = 100


class AssetManager(ABC):
    @abstractmethod
    def store(self, asset_guid: str, file_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, asset_guid: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def exists(self, asset_guid: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove(self, asset_guid: str) -> None:
        raise NotImplementedError


class LocalFileAssetManager(AssetManager):
    def __init__(self, local_storage_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_path = local_storage_path

    def store(self, asset_guid: str, src_file_ref: str) -> str:
        if self.exists(asset_guid):
            self.logger.info(f"Attempted to write repeated entry {asset_guid}.")
            return self.retrieve(asset_guid)

        dst_asset_dir = self._gen_path(asset_guid)
        try:
            os.makedirs(dst_asset_dir)
            filename = uri_to_filename(src_file_ref)
            dst_filename = os.path.join(dst_asset_dir, filename)
            if is_local(src_file_ref):
                shutil.copyfile(local_uri_to_path(src_file_ref), dst_filename)
            else:
                download_file(src_file_ref, dst_filename)
        except Exception:
            self.logger.exception(f"Exception when storing asset {src_file_ref}/{asset_guid}.")
            # Clean up asset directory
            try:
                shutil.rmtree(dst_asset_dir)
            except Exception:
                self.logger.exception(
                    "Exception when cleaning up directory after failing to "
                    f"store asset with ID {asset_guid}"
                )
                raise
            raise
        return dst_filename

    def retrieve(self, asset_guid: str) -> str:
        asset_path = self._gen_path(asset_guid)
        if not os.path.exists(asset_path):
            msg = f"File with ID {asset_guid} does not exist."
            self.logger.error(msg)
            raise ValueError(msg)
        files_in_asset_folder = os.listdir(asset_path)

        if len(files_in_asset_folder) != 1:
            msg = f"Inconsistent content found for asset ID {asset_guid}"
            self.logger.error(msg)
            raise ValueError(msg)

        file_name = files_in_asset_folder[0]
        return os.path.join(asset_path, file_name)

    def exists(self, asset_guid: str) -> bool:
        return os.path.exists(self._gen_path(asset_guid))

    def _gen_path(self, guid: str) -> str:
        return os.path.join(self.root_path, guid)

    def remove(self, asset_guid: str) -> None:
        if not self.exists(asset_guid):
            self.logger.info(f"Asked to remove inexistent file {asset_guid}.")
            return

        asset_folder = self._gen_path(asset_guid)

        try:
            shutil.rmtree(asset_folder)
        except Exception:
            msg = f"Could not remove asset with ID {asset_guid}"
            self.logger.exception(msg)
            raise ValueError(msg)


# ATTENTION: if the blob container associated with the assets is modified (through a write or
# delete) operation, then we should invalidate the cache of this function by calling its
# cache_clear() method.
@lru_cache(maxsize=CACHE_SIZE)
def cached_blob_list_by_prefix(client: ContainerClient, guid: str) -> List[BlobProperties]:
    return list(client.list_blobs(name_starts_with=guid))


class BlobServiceProvider(ABC):
    @abstractmethod
    def get_client(self) -> BlobServiceClient:
        raise NotImplementedError


class BlobServiceProviderWithCredentials(BlobServiceProvider):
    def __init__(
        self,
        storage_account_url: str,
        credential: Optional[TokenCredential] = None,
    ):
        self.credential = DefaultAzureCredential() if credential is None else credential
        self.client = BlobServiceClient(storage_account_url, self.credential)

    def get_client(self) -> BlobServiceClient:
        return self.client


class BlobServiceProviderWithConnectionString(BlobServiceProvider):
    def __init__(self, connection_string: str):
        self.client = BlobServiceClient.from_connection_string(connection_string)

    def get_client(self) -> BlobServiceClient:
        return self.client


class BlobAssetManager(AssetManager):
    blob_delimiter = "/"

    def __init__(
        self,
        storage_account_url: str = "",
        storage_account_connection_string: str = "",
        asset_container_name: str = DEFAULT_BLOB_ASSET_MANAGER_CONTAINER,
        credential: Optional[TokenCredential] = None,
        max_upload_concurrency: int = 6,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Create a blob client, authenticated.
        self.credential = DefaultAzureCredential() if credential is None else credential
        if storage_account_url:
            self.client = BlobServiceProviderWithCredentials(
                storage_account_url=storage_account_url, credential=self.credential
            ).get_client()
            self.blob_token_manager = BlobTokenManagerCredentialed(credential=self.credential)
        elif storage_account_connection_string:
            self.client = BlobServiceProviderWithConnectionString(
                connection_string=storage_account_connection_string
            ).get_client()
            self.blob_token_manager = BlobTokenManagerConnectionString(
                connection_string=storage_account_connection_string
            )
        else:
            msg = (
                "Could not get a blob manager since neither storage account "
                "url nor connection string were provided"
            )
            self.logger.exception(msg)
            raise ValueError(msg)

        self.container_name = asset_container_name
        self.container = self._retrieve_container()
        self.max_upload_concurrency = max_upload_concurrency

    def _retrieve_container(self):
        container = self.client.get_container_client(self.container_name)
        if not container.exists():
            container.create_container()

        return container

    @staticmethod
    def _join(*args: str):
        return BlobAssetManager.blob_delimiter.join(args)

    def _list(self, guid: str) -> List[BlobProperties]:
        listed_blob = cached_blob_list_by_prefix(self.container, guid)
        if len(listed_blob) > 1:
            ValueError(f"Encountered more than one asset with id {guid}")

        return listed_blob

    def _local_upload(self, file_path: str, blob_client: BlobClient):
        # At this point, we expect a valid local path was passed to the file_path
        # which can be something like "file:///path/to/file" or "/path/to/file".
        local_upload(file_path, blob_client, max_concurrency=self.max_upload_concurrency)

    def store(self, asset_guid: str, file_ref: str) -> str:
        if self.exists(asset_guid):
            self.logger.debug(f"Attempted to write repeated entry {asset_guid}.")
            blob_property = self._list(asset_guid)[0]
            blob_client = self.container.get_blob_client(blob_property.name)
            return blob_client.url

        filename = uri_to_filename(file_ref)
        blob_name = self._join(asset_guid, filename)
        blob_client = self.container.get_blob_client(blob_name)

        if is_local(file_ref):
            upload = self._local_upload
        else:
            upload = remote_upload

        try:
            upload(file_ref, blob_client)
        except Exception:
            self.logger.exception(f"Exception when storing asset {file_ref}/ ID {asset_guid}.")
            raise

        # Clear cache as we know we have modified the blob content
        cached_blob_list_by_prefix.cache_clear()

        return blob_client.url

    def retrieve(self, asset_guid: str) -> str:
        # Obtains a SAS token for file and creates a URL for it.
        if not self.exists(asset_guid):
            msg = f"File with ID {asset_guid} does not exist."
            self.logger.error(msg)
            raise ValueError(msg)

        blob_property = self._list(asset_guid)[0]
        blob_client = self.container.get_blob_client(blob_property.name)

        return self.blob_token_manager.sign_url(blob_client.url)

    def exists(self, asset_guid: str) -> bool:
        listed_blob = self._list(asset_guid)
        return len(listed_blob) == 1

    def remove(self, asset_guid: str) -> None:
        if not self.exists(asset_guid):
            self.logger.debug(f"Asked to remove inexistent file {asset_guid}.")
            return

        blob_property = self._list(asset_guid)[0]
        try:
            self.container.delete_blob(blob_property.name)
        except Exception:
            msg = f"Could not remove asset with ID {asset_guid}"
            self.logger.exception(msg)
            raise ValueError(msg)

        cached_blob_list_by_prefix.cache_clear()


AssetManagerConfig = builds(
    AssetManager,
    zen_dataclass={
        "module": "vibe_agent.storage.asset_management",
        "cls_name": "AssetManagerConfig",
    },
)


@dataclass
class BlobAssetManagerConfig(AssetManagerConfig):
    _target_: str = "vibe_agent.storage.asset_management.BlobAssetManager"
    storage_account_url: Any = MISSING
    storage_account_connection_string: Any = MISSING
    asset_container_name: Any = MISSING
    credential: Any = MISSING
    max_upload_concurrency: Any = 6


LocalFileAssetManagerConfig = builds(
    LocalFileAssetManager,
    populate_full_signature=True,
    builds_bases=(AssetManagerConfig,),
    zen_dataclass={
        "module": "vibe_agent.storage.asset_management",
        "cls_name": "LocalFileAssetManagerConfig",
    },
)
