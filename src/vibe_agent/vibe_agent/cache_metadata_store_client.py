import logging

from vibe_common.constants import DATA_OPS_INVOKE_URL_TEMPLATE
from vibe_common.schemas import OpRunId, OpRunIdDict
from vibe_common.telemetry import get_current_trace_parent
from vibe_common.vibe_dapr_client import VibeDaprClient
from vibe_core.data.core_types import OpIOType


class CacheMetadataStoreClient:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initializing CacheMetadataStoreClient")
        self.vibe_dapr_client = VibeDaprClient()

    async def add_refs(
        self,
        run_id: str,
        op_run_id: OpRunId,
        output: OpIOType,
    ) -> None:
        self.logger.debug(
            f"Adding refs for run {run_id} with op name = {op_run_id.name} "
            f"op hash = {op_run_id.hash}"
        )

        # Under load, Pydantic is having issues serializing the OpRunId dataclass object
        op_run_id_dict = OpRunIdDict(name=op_run_id.name, hash=op_run_id.hash)
        response = await self.vibe_dapr_client.post(
            url=DATA_OPS_INVOKE_URL_TEMPLATE.format("add_refs", run_id),
            data={
                "op_run_id_dict": self.vibe_dapr_client.obj_json(op_run_id_dict),
                "output": self.vibe_dapr_client.obj_json(output),
            },
            traceparent=get_current_trace_parent(),
        )

        assert response.ok, "Failed to add refs, but underlying method didn't capture it"
