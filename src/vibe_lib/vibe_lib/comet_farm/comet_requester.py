# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from queue import Queue
from typing import Any, Dict, Optional

import xmltodict
from pyngrok import ngrok

from vibe_lib.comet_farm.comet_model import CometOutput, CometResponse
from vibe_lib.comet_farm.comet_server import CometHTTPServer, CometServerParameters

TIMEOUT_IN_SECONDS = 120


class CometRequester:
    def __init__(self, comet_request: CometServerParameters):
        self.comet_request = comet_request

    def get_comet_raw_output(self, queue: "Queue[str]") -> str:
        return queue.get(timeout=TIMEOUT_IN_SECONDS * 60)

    def parse_comet_response(self, raw_comet_response: str) -> Dict[str, Any]:
        comet_xml = xmltodict.parse(raw_comet_response)
        comet_json = json.loads(json.dumps(comet_xml))
        return comet_json

    def run_comet_request(self, request_str: str) -> str:
        queue: "Queue[str]" = Queue()
        server = CometHTTPServer(queue, self.comet_request, request_str)
        comet_response = ""
        try:
            server.start()
            comet_response = self.get_comet_raw_output(queue)
            comet_json = self.parse_comet_response(comet_response)

            carbon_offset: Optional[str] = None
            # deriving the carbon offset
            cr = CometResponse(**comet_json)
            cLand = cr.day.cropland
            for scenario in cLand.modelRun.scenario:
                if type(scenario) == CometOutput and "scenario" in scenario.name:
                    co = CometOutput(**scenario.dict())
                    carbon_offset = co.carbon.soilCarbon + " Mg Co2e/year"
                    break

            if carbon_offset is None:
                raise RuntimeError("Missing carbon offset from COMET-Farm API")

            return carbon_offset
        except Exception as err:
            raise RuntimeError(
                f"Error when building comet response. Comet Response: {comet_response}"
            ) from err
        finally:
            server.shutdown()
            try:
                ngrok.kill()
            except Exception:
                pass
