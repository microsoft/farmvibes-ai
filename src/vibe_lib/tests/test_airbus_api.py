from typing import Dict
from unittest.mock import Mock, patch

import pytest

from vibe_lib.airbus import AirBusAPI, Constellation


@pytest.fixture(scope="module")
def api():
    with patch("vibe_lib.airbus.AirBusAPI._get_api_key") as mock_key:
        with patch("vibe_lib.airbus.AirBusAPI._authenticate") as mock_token:
            mock_key.return_value = "mock_api_key"
            mock_token.return_value = "mock_token"
            yield AirBusAPI("mock_filepath", False, [Constellation.PHR], 0.1, 0.4)


@pytest.fixture
def ordered_status():
    return {"id": "0", "status": "ordered"}


@pytest.fixture
def delivered_status():
    return {"id": "0", "status": "delivered"}


@pytest.fixture
def unkown_status():
    return {"id": "0", "status": "unknown"}


@patch("vibe_lib.airbus.AirBusAPI.get_order_by_id")
def test_ok_order(mock_handle: Mock, api: AirBusAPI, delivered_status: Dict[str, str]):
    mock_handle.return_value = delivered_status
    api.block_until_order_delivered("order_id")
    mock_handle.assert_called_once_with("order_id")


@patch("vibe_lib.airbus.AirBusAPI.get_order_by_id")
def test_unexpected_order_status(mock_handle: Mock, api: AirBusAPI, unkown_status: Dict[str, str]):
    mock_handle.return_value = unkown_status
    with pytest.raises(ValueError):
        api.block_until_order_delivered("order_id")


@patch("vibe_lib.airbus.AirBusAPI.get_order_by_id")
def test_timeout(mock_handle: Mock, api: AirBusAPI, ordered_status: Dict[str, str]):
    mock_handle.return_value = ordered_status
    with pytest.raises(RuntimeError):
        api.block_until_order_delivered("order_id")
    assert mock_handle.call_count == 5
