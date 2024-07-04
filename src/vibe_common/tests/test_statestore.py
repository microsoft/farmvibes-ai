from typing import Any

import pytest

from vibe_common.statestore import StateStore


class MockResponse:
    def __init__(self, content: Any):
        self._content = content

    async def json(self, loads: Any, **kwargs: Any) -> Any:
        return loads(self._content, **kwargs)


@pytest.mark.anyio
async def test_store_fails_with_invalid_input():
    store = StateStore()
    for value in [float(x) for x in "inf -inf nan".split()]:
        with pytest.raises(ValueError):
            await store.store("key", value)
