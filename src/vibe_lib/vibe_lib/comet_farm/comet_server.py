# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import logging
import os
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Any, Optional, cast

import requests
from pydantic.main import BaseModel
from pyngrok import conf, ngrok

HTTP_SERVER_PORT: int = 1108
HTTP_SERVER_HOST: str = "0.0.0.0"


class CometServerParameters(BaseModel):
    url: str
    webhook: str
    ngrokToken: str
    supportEmail: str
    apiKey: str


class CometHTTPServer(Thread):
    def __init__(
        self, outqueue: "Queue[str]", comet_request: CometServerParameters, request_str: str
    ):
        def handler(*args: Any, **kwargs: Any):
            return CometHTTPRequestHandler(outqueue, *args, **kwargs)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.outqueue = outqueue
        self.comet_request = comet_request
        self.ngrok_token = comet_request.ngrokToken
        self.server = HTTPServer((HTTP_SERVER_HOST, HTTP_SERVER_PORT), handler)
        self.tunnel: Optional[Any] = None
        self.tmpdir = TemporaryDirectory()
        self.ngrok_config = conf.get_default()
        self.ngrok_config.ngrok_path = os.path.join(self.tmpdir.name, "ngrok")  # type: ignore
        self.started_server = False
        self.request_str = request_str

        super().__init__()

    def start_ngrok(self):
        ngrok.set_auth_token(self.ngrok_token, self.ngrok_config)
        self.tunnel = ngrok.connect(HTTP_SERVER_PORT, bind_tls=True)
        self.comet_request.webhook = self.tunnel.public_url

    def submit_job(self, xml_string: str, reference_id: str = ""):
        xml_file = io.StringIO(xml_string)
        postUrl = self.comet_request.url
        webhookUrl = self.comet_request.webhook + "/" + reference_id

        payload = {
            "LastCropland": "-1",
            "FirstCropland": "-1",
            "email": self.comet_request.supportEmail,
            "url": webhookUrl,
            "LastDaycentInput": "0",
            "FirstDaycentInput": "0",
            "apikey": self.comet_request.apiKey
        }

        files = {"file": ("file.xml", xml_file, "application/xml")}
        headers = {}

        self.logger.info(f"Submitting {payload} to COMET-Farm API")
        r = requests.request("POST", postUrl, headers=headers, data=payload, files=files)

        # raise exception on error
        r.raise_for_status()

        return r.text

    def run(self):
        try:
            self.start_ngrok()
            request_id = str(uuid.uuid4())
            self.submit_job(self.request_str, reference_id=request_id)
            self.started_server = True
            self.server.serve_forever()
        except Exception:
            self.outqueue.put(f"Failed to submit job to COMET-Farm API: {traceback.format_exc()}")
            raise

    def shutdown(self):
        if self.started_server:
            self.server.shutdown()
        if self.tunnel is not None:
            ngrok.disconnect(self.tunnel.public_url)
        self.tmpdir.cleanup()


class CometHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, outqueue: "Queue[str]", *args: Any, **kwargs: Any):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.outqueue = outqueue
        super().__init__(*args, **kwargs)

    def _send_ok(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        content_len_str = cast(str, self.headers.get("Content-Length"))
        content_len = int(content_len_str, 0)
        post_body = self.rfile.read(content_len).decode("utf-8")
        self.logger.info(f"Received data {post_body} from COMET-Farm API")
        self.outqueue.put(post_body)
        self._send_ok()

    def do_GET(self):
        self._send_ok()
