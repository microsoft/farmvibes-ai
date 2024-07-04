import json
import logging
import os
import shutil
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Union

from azure.identity import AzureCliCredential
from hydra_zen import builds
from pystac.item import Item

from vibe_agent.ops import (
    BaseVibeDict,
    Operation,
    OperationFactory,
    OperationFactoryConfig,
    OperationSpec,
    OpIOType,
    OpResolvedDependencies,
    TypeDictVibe,
)
from vibe_agent.ops_helper import OpIOConverter
from vibe_agent.storage import Storage
from vibe_agent.storage.asset_management import LocalFileAssetManager
from vibe_agent.storage.storage import ItemDict, ensure_list
from vibe_common.schemas import CacheInfo, OperationParser, OpRunId
from vibe_common.secret_provider import AzureSecretProvider, SecretProvider
from vibe_core import data
from vibe_core.data.core_types import BaseVibe
from vibe_core.data.json_converter import DataclassJSONEncoder
from vibe_core.data.utils import deserialize_stac, serialize_stac

LOGGER = logging.getLogger(__name__)
REFERENCE_FILENAME = "reference.json"


class FakeStorage(Storage):
    def store(self, items: List[Item]) -> List[Item]:
        return items

    def retrieve(self, input_item_dicts: List[Item]) -> List[Item]:
        return input_item_dicts

    def retrieve_output_from_input_if_exists(self, input_item: Item) -> Optional[Item]:
        return input_item

    async def retrieve_output_from_input_if_exists_async(
        self, cache_info: CacheInfo, **kwargs: Any
    ) -> Optional[ItemDict]:
        raise NotImplementedError

    def remove(self, op_run_id: OpRunId):
        return None


class OpTester:
    def __init__(self, path_to_config: str):
        self._tmp_dir = TemporaryDirectory()
        self.asset_manager = LocalFileAssetManager(self._tmp_dir.name)
        self.fake_storage = FakeStorage(self.asset_manager)
        self.spec = OperationParser.parse(path_to_config)

    def run(self, **input_dict: Union[BaseVibe, List[BaseVibe]]) -> BaseVibeDict:
        self.op = OperationFactory(
            self.fake_storage, AzureSecretProvider(credential=AzureCliCredential())
        ).build(self.spec)
        return self.op.callback(**input_dict)

    def update_parameters(self, parameters: Dict[str, Any]):
        self.spec.parameters.update(parameters)

    def __del__(self):
        try:
            self._tmp_dir.cleanup()
        except (AttributeError, FileNotFoundError):
            LOGGER.info(f"Unable to clean temporary directory {self._tmp_dir}")


class ReferenceSaver(Operation):
    storage: Storage

    def __init__(
        self,
        name: str,
        callback: Callable[..., BaseVibeDict],
        storage: Storage,
        converter: data.StacConverter,
        inputs_spec: TypeDictVibe,
        output_spec: TypeDictVibe,
        version: str,
        dependencies: OpResolvedDependencies,
        save_dir: str,
    ):
        self.root_dir = save_dir
        self.dependencies = dependencies
        super().__init__(name, callback, storage, converter, inputs_spec, output_spec, version)

    def _get_ref_path(self) -> str:
        return os.path.join(self.root_dir, self.name, REFERENCE_FILENAME)

    def _get_reference(self) -> List[Any]:
        ref_path = self._get_ref_path()
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return json.load(f)
        return []

    def _update_reference(self, stac_inputs: ItemDict, stac_outputs: ItemDict):
        ref = self._get_reference()
        serialized = [
            {k: serialize_stac(v) for k, v in s.items()} for s in (stac_inputs, stac_outputs)
        ]
        ref.append(serialized)
        with open(self._get_ref_path(), "w") as f:
            json.dump(ref, f, cls=DataclassJSONEncoder)

    def save_items(self, items: ItemDict) -> ItemDict:
        save_items = deepcopy(items)
        for item_list in save_items.values():
            item_list = ensure_list(item_list)
            for item in item_list:
                for k, v in item.assets.items():
                    rel_path = os.path.join(self.name, k)
                    abs_path = os.path.join(self.root_dir, rel_path)
                    filepath = self.storage.asset_manager.retrieve(k)
                    try:
                        os.makedirs(abs_path)
                        shutil.copy(filepath, abs_path)
                    except FileExistsError:
                        # File exists so we don't copy again
                        pass
                    v.href = os.path.join(rel_path, os.path.basename(filepath))

        return save_items

    def save_inputs_and_outputs(self, input_items: ItemDict, output_items: ItemDict):
        save_inputs = self.save_items(input_items)
        save_outputs = self.save_items(output_items)
        self._update_reference(save_inputs, save_outputs)

    def run(self, input_items: OpIOType) -> OpIOType:
        stac_inputs = OpIOConverter.deserialize_input(input_items)
        cache_info = CacheInfo(self.name, self.version, stac_inputs, self.dependencies)
        items_out = super().run(input_items, cache_info)
        stac_outputs = {k: deserialize_stac(v) for k, v in items_out.items()}
        # Create directory for the op if necessary
        os.makedirs(os.path.join(self.root_dir, self.name), exist_ok=True)
        self.save_inputs_and_outputs(stac_inputs, stac_outputs)
        return items_out


class ReferenceSaverFactory(OperationFactory):
    storage: Storage
    save_dir: str

    def __init__(self, storage: Storage, secret_provider: SecretProvider, save_dir: str):
        super().__init__(storage, secret_provider)
        self.save_dir = save_dir

    def _build_impl(self, op_config: OperationSpec) -> ReferenceSaver:
        parameters = self.resolve_secrets(op_config.parameters)
        dependencies = self.dependency_resolver.resolve(op_config)
        callable = self.callable_builder.build(
            op_config.root_folder, op_config.entrypoint, parameters
        )
        return ReferenceSaver(
            op_config.name,
            callable,
            self.storage,
            self.converter,
            op_config.inputs_spec,
            op_config.output_spec,
            op_config.version,
            dependencies,
            self.save_dir,
        )


ReferenceSaverFactoryConfig = builds(
    ReferenceSaverFactory,
    save_dir=str,
    builds_bases=(OperationFactoryConfig,),
)


class ReferenceRetriever:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.converter = data.StacConverter()

    def retrieve_assets(self, items: Union[List[Item], Item]):
        item: Item
        for item in ensure_list(items):
            for asset in item.assets.values():
                asset.href = os.path.join(self.root_dir, asset.href)

    def retrieve_op_io(self, item_dict: OpIOType) -> ItemDict:
        stac_items = {k: deserialize_stac(v) for k, v in item_dict.items()}
        for items in stac_items.values():
            self.retrieve_assets(items)
        return stac_items

    def to_terravibes(self, item_dict: ItemDict) -> BaseVibeDict:
        return {k: self.converter.from_stac_item(v) for k, v in item_dict.items()}

    def retrieve(self, op_name: str) -> List[List[BaseVibeDict]]:
        op_dir = os.path.join(self.root_dir, op_name)
        with open(os.path.join(op_dir, REFERENCE_FILENAME)) as f:
            pairs = json.load(f)
        stac_pairs = [[self.retrieve_op_io(i) for i in pair] for pair in pairs]
        output_pairs = [[self.to_terravibes(i) for i in pair] for pair in stac_pairs]
        return output_pairs
