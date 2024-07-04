import importlib.util
import inspect
import logging
import os
from importlib.abc import Loader
from typing import Any, Callable, Dict, List, Optional, Union

from azure.cosmos.exceptions import CosmosResourceExistsError
from hydra_zen import builds

from vibe_agent.ops_helper import OpIOConverter
from vibe_agent.storage.local_storage import LocalResourceExistsError
from vibe_common.schemas import (
    CacheInfo,
    EntryPointDict,
    ItemDict,
    OperationParser,
    OperationSpec,
    OpResolvedDependencies,
)
from vibe_common.secret_provider import SecretProvider, SecretProviderConfig
from vibe_core import data
from vibe_core.data.core_types import BaseVibeDict, InnerIOType, OpIOType, TypeDictVibe

from .storage import Storage, StorageConfig


class Operation:
    name: str
    callback: Callable[..., BaseVibeDict]
    storage: Storage
    converter: data.StacConverter
    inputs_spec: TypeDictVibe
    output_spec: TypeDictVibe
    version: str

    def __init__(
        self,
        name: str,
        callback: Callable[..., BaseVibeDict],
        storage: Storage,
        converter: data.StacConverter,
        inputs_spec: TypeDictVibe,
        output_spec: TypeDictVibe,
        version: str,
    ):
        self.name = name
        self.callback = callback
        self.storage = storage
        self.converter = converter
        self.inputs_spec = inputs_spec
        self.output_spec = output_spec
        self.version = version
        self.logger = logging.getLogger(self.__class__.__name__)

        intersection = set(inputs_spec.keys()).intersection(output_spec.keys())
        if intersection:
            raise ValueError(
                f"Operation {name} has input and output with conflicting names {intersection}"
            )

    def _fetch_from_cache(self, cache_info: CacheInfo) -> Optional[OpIOType]:
        """
        Try to fetch output from the cache, returns `None` if no output is found
        """
        items = self.storage.retrieve_output_from_input_if_exists(cache_info)
        if items is not None:
            items = OpIOConverter.serialize_output(items)
        return items

    def _call_validate_op(self, **kwargs: InnerIOType) -> ItemDict:
        results = self.callback(**kwargs)
        result_keys = set(results)
        output_keys = set(self.output_spec)
        if result_keys != output_keys:
            raise RuntimeError(
                f"Invalid output obtained during execution of op '{self.name}'. "
                f"Expected output keys {output_keys}, but callback returned {result_keys}"
            )
        try:
            return {k: self.converter.to_stac_item(v) for k, v in results.items()}
        except AttributeError:
            raise ValueError(
                f"Expected a dict-like as return value of operation {self.name}, found "
                f"{type(results)}"
            )

    # Run will run the operation, loading the data from the catalog
    def run(self, input_items: OpIOType, cache_info: CacheInfo) -> OpIOType:
        stac_items = OpIOConverter.deserialize_input(input_items)
        op_hash = cache_info.hash
        items_out = self._fetch_from_cache(cache_info)
        if items_out is not None:
            self.logger.warning(
                f"Cache hit for op {self.name} with cache hash {op_hash} before computation, "
                "probably due to a repeated message."
            )
            return items_out

        self.logger.info(f"Running op {self.name} for cache hash {op_hash}")
        run_id = data.gen_guid()
        retrieved_items = self.storage.retrieve(stac_items)
        self.logger.info(f"Retrieved input for op {self.name}")
        items = {k: self.converter.from_stac_item(v) for k, v in retrieved_items.items()}
        self.logger.info(f"Running callback for op {self.name}")
        stac_results = self._call_validate_op(**items)
        self.logger.info(f"Callback finished for op {self.name}")

        try:
            items_out = self.storage.store(run_id, stac_results, cache_info)
            self.logger.info(f"Output stored for op {self.name}")
        except (LocalResourceExistsError, CosmosResourceExistsError):
            # If two instances of the same op with the same input start running at the same time
            # We'll have a race condition where they'll both run, and try to store into the cache
            # This will instead retrieve the output from the op that wrote their results first
            items_out = self._fetch_from_cache(cache_info)
            if items_out is not None:
                self.logger.warning(
                    f"Cache hit after computing op {self.name} with cache hash {op_hash}, "
                    "probably due to a race condition."
                )
                return items_out
            raise  # We couldn't write and we can't read, so we break

        return OpIOConverter.serialize_output(items_out)


class CallableBuilder:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _resolve_callable(
        self, op_root_folder: str, filename: str, callback_builder_name: str
    ) -> Any:
        modname = os.path.splitext(filename)[0]
        path = os.path.join(op_root_folder, filename)
        self.logger.debug(
            f"Loading module spec for {modname} from path {path} "
            f"with callback {callback_builder_name}"
        )
        spec = importlib.util.spec_from_file_location(modname, path)
        assert spec is not None
        self.logger.debug(f"Loading module {modname} from spec")
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, Loader)
        self.logger.debug(f"Executing module {modname}")
        spec.loader.exec_module(module)
        self.logger.debug(f"Getting callback {callback_builder_name} from module {modname}")
        callback_builder = getattr(module, callback_builder_name)

        return callback_builder

    def build(
        self,
        op_root_folder: str,
        entrypoint: EntryPointDict,
        parameters: Dict[str, Any],
    ) -> Callable[[Any], Any]:
        self.logger.debug(f"Building callable builder for {entrypoint}")
        callable_builder = self._resolve_callable(
            op_root_folder,
            entrypoint["file"],
            entrypoint["callback_builder"],
        )
        self.logger.debug(f"Building callable from {callable_builder}")
        callable = callable_builder(**parameters)
        if inspect.isclass(callable_builder):
            callable = callable()
        self.logger.debug(f"Built callable {callable}")
        return callable


class OperationDependencyResolver:
    def __init__(self):
        self._resolver_map = {"parameters": self._resolve_params}

    def resolve(self, op_spec: OperationSpec) -> OpResolvedDependencies:
        output: OpResolvedDependencies = {}
        for item, dependencies_list in op_spec.dependencies.items():
            try:
                output[item] = self._resolver_map[item](op_spec, dependencies_list)
            except Exception as e:
                raise ValueError(
                    f"Dependency {item}: {dependencies_list} could not be resolved"
                ) from e
        return output

    def _resolve_params(self, op_spec: OperationSpec, params_to_resolve: List[str]):
        return {param_name: op_spec.parameters[param_name] for param_name in params_to_resolve}


class OperationFactory:
    converter: data.StacConverter
    storage: Storage
    secret_provider: SecretProvider
    callable_builder: CallableBuilder
    dependency_resolver: OperationDependencyResolver

    def __init__(self, storage: Storage, secret_provider: SecretProvider):
        self.storage = storage
        self.converter = data.StacConverter()
        self.callable_builder = CallableBuilder()
        self.secret_provider = secret_provider

        self.dependency_resolver = OperationDependencyResolver()

    def build(self, op_definition: Union[str, OperationSpec]) -> Operation:
        if isinstance(op_definition, str):
            return self._build_impl(OperationParser.parse(op_definition))
        return self._build_impl(op_definition)

    def resolve_secrets(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self.secret_provider.resolve(v) for k, v in parameters.items()}

    def _build_impl(self, op_config: OperationSpec) -> Operation:
        parameters = self.resolve_secrets(op_config.parameters)
        callable = self.callable_builder.build(
            op_config.root_folder, op_config.entrypoint, parameters
        )

        return Operation(
            op_config.name,
            callable,
            self.storage,
            self.converter,
            op_config.inputs_spec,
            op_config.output_spec,
            op_config.version,
        )


OperationFactoryConfig = builds(
    OperationFactory,
    storage=StorageConfig,
    secret_provider=SecretProviderConfig,
    zen_dataclass={"module": "vibe_agent.ops", "cls_name": "OperationFactoryConfig"},
)
