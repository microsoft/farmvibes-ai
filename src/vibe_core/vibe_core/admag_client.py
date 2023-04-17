import json
from typing import Any, Dict, List, cast
from urllib.parse import urljoin

import msal
import requests
from requests.exceptions import HTTPError


class ADMAgClient:
    """Client for Azure Data Manager for Agriculture (ADMAg) API.

    :param base_url: The base URL for the ADMAg API.

    :param api_version: The API version to be used.

    :param client_id: The client ID.

    :param client_secret: The client secret.

    :param authority: The URI of the identity provider.

    :param default_scope: The scope of the access request.
    """

    DEFAULT_TIMEOUT = 120
    """Default timeout for requests."""

    NEXT_PAGES_LIMIT = 100000
    """Maximum number of pages to retrieve in a single request."""

    CONTENT_TAG = "value"
    """Tag for the content of the response."""

    LINK_TAG = "nextLink"
    """Tag for the next link of the response."""

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

    def get_token(self, client_id: str, client_secret: str, authority: str, default_scope: str):
        """
        Generates the ADMAg access token to be used before each call.

        :param client_id: The client ID.

        :param client_secret: The client secret.

        :param authority: The URI of the identity provider.

        :param default_scope: The scope of the access request.

        :return: The access token as a string.

        :raises Exception: If error retrieving token.
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
        """Generates a header containing authorization for ADMAg API requests.

        :return: A dictionary containing the header for the API requests.
        """
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
                    raise RuntimeError(f"Repeated nextLink {next_link} in ADMAg get request")

                if next_link_index >= self.NEXT_PAGES_LIMIT:
                    raise RuntimeError(f"Next pages limit {self.NEXT_PAGES_LIMIT} exceded")
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
        """Retrieves the seasonal fields for a given farmer.

        :param farmer_id: The ID of the farmer.

        :param params: Additional parameters to be passed to the request. Defaults to {}.

        :return: The information for each seasonal fields.
        """
        endpoint = f"/farmers/{farmer_id}/seasonal-fields"
        request_params = {"api-version": self.api_version}
        request_params.update(params)

        return self._get(
            endpoint=endpoint,
            params=request_params,
        )

    def get_field(self, farmer_id: str, field_id: str):
        """
        Retrieves the field information for a given farmer and field.

        :param farmer_id: The ID of the farmer.

        :param field_id: The ID of the field.

        :return: The field information.
        """
        endpoint = f"/farmers/{farmer_id}/fields/{field_id}"
        return self._get(endpoint)

    def get_seasonal_field(self, farmer_id: str, seasonal_field_id: str):
        """Retrieves the information of a seasonal field for a given farmer.

        :param farmer_id: The ID of the farmer.

        :param seasonal_field_id: The ID of the seasonal field.

        :return: The seasonal field information.
        """
        endpoint = f"/farmers/{farmer_id}/seasonal-fields/{seasonal_field_id}"
        return self._get(endpoint)

    def get_boundary(self, farmer_id: str, boundary_id: str):
        """Retrieves the information of a boundary for a given farmer.

        :param farmer_id: The ID of the farmer.

        :param boundary_id: The ID of the boundary.

        :return: The boundary information.
        """
        endpoint = f"farmers/{farmer_id}/boundaries/{boundary_id}"
        return self._get(endpoint)

    def get_season(self, season_id: str):
        """Retrieves season information with a given id.

        :param season_id: The id of the season to retrieve.

        :return: The season data.
        """
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
        """
        Retrieves the information of a specified operation for a given farmer.

        This method will return information about the specified operation name,
        in the specified time range, for the given farmer and associated boundary IDs.

        :param farmer_id: The ID of the farmer.

        :param associated_boundary_ids: The IDs of the boundaries associated to the operation.

        :param operation_name: The name of the operation.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :param sources: (optional) The sources of the operation.

        :return: The operation information.
        """
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
        """Retrieves the harvest information for a given farmer.

        This method will return the harvest information for a given farmer,
        associated with the provided boundary ids, between the start and end
        operation dates specified.

        :param farmer_id: ID of the farmer.

        :param associated_boundary_ids: List of associated boundary IDs.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with harvest information.
        """
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
        """Retrieves the fertilizer information for a given farmer.

        This method will return the fertilizer information for a given farmer,
        associated with the provided boundary ids, between the start and end
        operation dates specified.

        :param farmer_id: ID of the farmer.

        :param associated_boundary_ids: List of associated boundary IDs.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with fertilizer information.
        """
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
        """Retrieves the organic amendments information for a given farmer.

        This method will return the organic amendments information for a given farmer,
        associated with the provided boundary ids, between the start and end
        operation dates specified.

        :param farmer_id: ID of the farmer.

        :param associated_boundary_ids: List of associated boundary IDs.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with organic amendments information.
        """

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
        """Retrieves the tillage information for a given farmer.

        This method will return the tillage information for a given farmer,
        associated with the provided boundary ids, between the start and end
        operation dates specified.

        :param farmer_id: ID of the farmer.

        :param associated_boundary_ids: List of associated boundary IDs.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with tillage information.
        """
        return self.get_operation_info(
            farmer_id=farmer_id,
            associated_boundary_ids=associated_boundary_ids,
            operation_name="tillage-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
        )
