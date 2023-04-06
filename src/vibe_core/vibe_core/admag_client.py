import json
from typing import Any, Dict, List, cast
from urllib.parse import urljoin

import msal
import requests
from requests.exceptions import HTTPError


class ADMAgClient:

    DEFAULT_TIMEOUT = 120
    NEXT_PAGES_LIMIT = 100000
    CONTENT_TAG = "value"
    LINK_TAG = "nextLink"

    def __init__(
        self,
        base_url: str,
        api_version: str,
        client_id: str,
        client_secret: str,
        authority: str,
        default_scope: str,
    ):
        self.token = self.get_token(
            client_id=client_id,
            client_secret=client_secret,
            authority=authority,
            default_scope=default_scope,
        )
        self.api_version = api_version
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(self.header())

    def get_token(
        self, client_id: str, client_secret: str, authority: str, default_scope: str
    ):
        """
        This method returns the access token as a string.
        Use this to fetch access token before each call.
        """
        app = msal.ConfidentialClientApplication(
            client_id=client_id, client_credential=client_secret, authority=authority
        )

        # Initialize the token to access admag resources
        # default value is none, if token for application is alread initialized
        app.acquire_token_silent(scopes=[default_scope], account=None)

        token_result = cast(Dict[str, Any], app.acquire_token_for_client(scopes=default_scope))
        if "access_token" in token_result:
            return token_result["access_token"]
        else:
            message = {
                "error": token_result.get("error"),
                "description": token_result.get("error_description"),
                "correlationId": token_result.get("correlation_id"),
            }

            raise Exception(message)

    def header(self) -> Dict[str, str]:
        header: Dict[str, str] = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/merge-patch+json",
        }

        return header

    def _request(self, method: str, endpoint: str, *args: Any, **kwargs: Any):
        response = self.session.request(method, urljoin(self.base_url, endpoint), *args, **kwargs)
        try:
            r = json.loads(response.text)
        except json.JSONDecodeError:
            r = response.text
        try:
            response.raise_for_status()
        except HTTPError as e:
            error_message = r.get("message", "") if isinstance(r, dict) else r
            msg = f"{e}. {error_message}"
            raise HTTPError(msg, response=e.response)
        return cast(Any, r)

    def _get(self, endpoint: str, params: Dict[str, Any] = {}):
        request_params = {"api-version": self.api_version}
        request_params.update(params)
        response = self._request(
            "GET",
            endpoint,
            params=request_params,
            timeout=self.DEFAULT_TIMEOUT,
        )
        visited_next_links = set()

        if self.CONTENT_TAG in response:
            composed_response = {self.CONTENT_TAG: response[self.CONTENT_TAG]}
            next_link = "" if self.LINK_TAG not in response else response[self.LINK_TAG]
            next_link_index = 0
            while next_link:
                if next_link in visited_next_links:
                    raise RuntimeError(
                        f"Repeated nextLink {next_link} in ADMAg get request"
                    )

                if next_link_index >= self.NEXT_PAGES_LIMIT:
                    raise RuntimeError(
                        f"Next pages limit {self.NEXT_PAGES_LIMIT} exceded"
                    )
                tmp_response = self._request(
                    "GET",
                    next_link,
                    timeout=self.DEFAULT_TIMEOUT,
                )
                if self.CONTENT_TAG in tmp_response:
                    composed_response[self.CONTENT_TAG].extend(tmp_response[self.CONTENT_TAG])
                visited_next_links.add(next_link)
                next_link_index = next_link_index + 1
                next_link = "" if self.LINK_TAG not in tmp_response else tmp_response[self.LINK_TAG]
            response = composed_response
        return response

    def get_seasonal_fields(self, farmer_id: str, params: Dict[str, Any] = {}):
        endpoint = f"/farmers/{farmer_id}/seasonal-fields"
        request_params = {"api-version": self.api_version}
        request_params.update(params)

        return self._get(
            endpoint=endpoint,
            params=request_params,
        )

    def get_field(self, farmer_id: str, field_id: str):
        endpoint = f"/farmers/{farmer_id}/fields/{field_id}"
        return self._get(endpoint)

    def get_seasonal_field(self, farmer_id: str, seasonal_field_id: str):
        endpoint = f"/farmers/{farmer_id}/seasonal-fields/{seasonal_field_id}"
        return self._get(endpoint)

    def get_boundary(self, farmer_id: str, boundary_id: str):
        endpoint = f"farmers/{farmer_id}/boundaries/{boundary_id}"
        return self._get(endpoint)

    def get_season(self, season_id: str):
        endpoint = f"/seasons/{season_id}"
        return self._get(endpoint)

    def get_operation_info(
        self,
        farmer_id: str,
        associated_boundary_ids: List[str],
        operation_name: str,
        min_start_operation: str,
        max_end_operation: str,
        sources: List[str] = [],
    ):
        endpoint = f"/farmers/{farmer_id}/{operation_name}"
        params = {
            "api-version": self.api_version,
            "associatedBoundaryIds": associated_boundary_ids,
            "minOperationStartDateTime": min_start_operation,
            "maxOperationEndDateTime": max_end_operation,
        }

        if sources:
            params["sources"] = sources

        return self._get(endpoint, params=params)

    def get_harvest_info(
        self,
        farmer_id: str,
        associated_boundary_ids: List[str],
        min_start_operation: str,
        max_end_operation: str,
    ):
        return self.get_operation_info(
            farmer_id=farmer_id,
            associated_boundary_ids=associated_boundary_ids,
            operation_name="harvest-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
        )

    def get_fertilizer_info(
        self,
        farmer_id: str,
        associated_boundary_ids: List[str],
        min_start_operation: str,
        max_end_operation: str,
    ):
        return self.get_operation_info(
            farmer_id=farmer_id,
            associated_boundary_ids=associated_boundary_ids,
            operation_name="application-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            sources=["Fertilizer"],
        )

    def get_organic_amendments_info(
        self,
        farmer_id: str,
        associated_boundary_ids: List[str],
        min_start_operation: str,
        max_end_operation: str,
    ):
        return self.get_operation_info(
            farmer_id=farmer_id,
            associated_boundary_ids=associated_boundary_ids,
            operation_name="application-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            sources=["Omad"],
        )

    def get_tillage_info(
        self,
        farmer_id: str,
        associated_boundary_ids: List[str],
        min_start_operation: str,
        max_end_operation: str,
    ):
        return self.get_operation_info(
            farmer_id=farmer_id,
            associated_boundary_ids=associated_boundary_ids,
            operation_name="tillage-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
        )
