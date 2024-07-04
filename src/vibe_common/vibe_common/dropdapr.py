"""
dropdapr - A drop-in replacement for dapr-ext-grpc subscribe using FastAPI.
"""

from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, TypedDict, Union

import uvicorn
from cloudevents.sdk.event import v1
from fastapi import FastAPI  # type: ignore
from pydantic import BaseConfig

BaseConfig.arbitrary_types_allowed = True


class TopicEventResponse(Dict[str, str]):
    def __getattr__(self, attr: str):
        if attr == "status":
            return self["status"]

    def __init__(self, *args: Any, **kwargs: Dict[Any, Any]):
        if len(args) == 1 and isinstance(args[0], str):
            super().__init__({"status": args[0].upper()})
        else:
            super().__init__(*args, **kwargs)


class TopicEventResponseStatus:
    success: TopicEventResponse = TopicEventResponse({"status": "SUCCESS"})
    retry: TopicEventResponse = TopicEventResponse({"status": "RETRY"})
    drop: TopicEventResponse = TopicEventResponse({"status": "DROP"})


class DaprSubscription(TypedDict):
    pubsubname: str
    topic: str
    route: str
    metadata: Optional[Dict[str, str]]


class App:
    def __init__(self):
        self.app = FastAPI()
        self.subscriptions: List[DaprSubscription] = []
        self.server: Optional[uvicorn.Server] = None

        self.app.add_api_route(
            "/",
            self.index,  # type: ignore
            methods=["GET"],
            response_model=Dict,
        )

        self.app.add_api_route(
            "/dapr/subscribe",
            lambda: self.subscriptions,  # type: ignore
            methods=["GET"],
            response_model=Any,
        )

    def index(self):
        return {
            "detail": "This server only works with dapr. Please don't make requests to it.",
            "subscriptions": self.subscriptions,
        }

    def add_subscription(
        self,
        handler: Callable[..., Union[TopicEventResponse, Coroutine[Any, Any, Any]]],
        pubsub: str,
        topic: str,
        metadata: Optional[Dict[str, str]] = {},
    ):
        event_handler_route = f"/events/{pubsub}/{topic}"
        self.app.add_api_route(
            event_handler_route,
            handler,  # type: ignore
            methods=["POST"],
            response_model=Any,
        )

        self.subscriptions.append(
            {
                "pubsubname": pubsub,
                "topic": topic,
                "route": event_handler_route,
                "metadata": metadata,
            }
        )

    def subscribe_async(self, pubsub: str, topic: str, metadata: Optional[Dict[str, str]] = {}):
        def decorator(func: Callable[[v1.Event], Awaitable[Any]]):
            async def event_wrapper(request: Dict[str, Any]):
                event = v1.Event()
                event.SetEventType(request["type"])
                event.SetEventID(request["id"])
                event.SetSource(request["source"])
                try:
                    event.SetData(request["data"])
                except KeyError:
                    event.SetData(request["data_base64"])
                event.SetContentType(request["datacontenttype"])
                try:
                    return await func(event)
                except RuntimeError:
                    return TopicEventResponseStatus.retry
                except Exception:
                    return TopicEventResponseStatus.drop

            self.add_subscription(event_wrapper, pubsub, topic, metadata)

        return decorator

    def subscribe(self, pubsub: str, topic: str, metadata: Optional[Dict[str, str]] = {}):
        def decorator(func: Callable[[v1.Event], Any]):
            def event_wrapper(request: Dict[str, Any]):
                event = v1.Event()
                event.SetEventType(request["type"])
                event.SetEventID(request["id"])
                event.SetSource(request["source"])
                try:
                    event.SetData(request["data"])
                except KeyError:
                    event.SetData(request["data_base64"])
                event.SetContentType(request["datacontenttype"])
                try:
                    return func(event)
                except RuntimeError:
                    return TopicEventResponseStatus.retry
                except Exception:
                    return TopicEventResponseStatus.drop

            self.add_subscription(event_wrapper, pubsub, topic, metadata)

        return decorator

    def method(self, name: str):
        def decorator(func):  # type: ignore
            route = f"/{name}"
            self.app.add_api_route(
                route,
                func,
                methods=["GET", "POST"],
                response_model=Any,
            )

        return decorator

    def startup(self):
        def decorator(func: Callable[[], None]):
            self.app.add_event_handler("startup", func)

        return decorator

    def shutdown(self):
        def decorator(func):  # type: ignore
            self.app.add_event_handler("shutdown", func)

        return decorator

    def health(self, endpoint: str = "/health"):
        def decorator(func):  # type: ignore
            self.app.add_api_route(
                endpoint,
                func,
                methods=["GET"],
                response_model=Any,
            )

        return decorator

    def run(
        self,
        port: int,
        limit_concurrency: Optional[int] = None,
    ):
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            limit_concurrency=limit_concurrency,
        )
        self.server = uvicorn.Server(config)
        self.server.run()  # type: ignore

    async def run_async(
        self,
        port: int,
        limit_concurrency: Optional[int] = None,
        workers: int = 1,
    ):
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            limit_concurrency=limit_concurrency,
            loop="uvloop",
            workers=workers,
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()
