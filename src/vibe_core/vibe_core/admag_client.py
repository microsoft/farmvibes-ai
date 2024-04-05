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

    def _request(
        self, method: str, endpoint: str, data: Dict[str, Any] = {}, *args: Any, **kwargs: Any
    ):
        resp = self.session.request(
            method, urljoin(self.base_url, endpoint), *args, **kwargs, json=data
        )
        try:
            r = json.loads(resp.text)
        except json.JSONDecodeError:
            r = resp.text
        try:
            resp.raise_for_status()
        except HTTPError as e:
            error_message = r.get("message", "") if isinstance(r, dict) else r
            msg = f"{e}. {error_message}"
            raise HTTPError(msg, response=e.response)

        return cast(Any, r)

    def _iterate(self, response: Dict[str, Any]):
        visited_next_links = set()

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

    def _get(self, endpoint: str, params: Dict[str, Any] = {}):
        request_params = {"api-version": self.api_version}
        request_params.update(params)
        response = self._request(
            "GET",
            endpoint,
            params=request_params,
            timeout=self.DEFAULT_TIMEOUT,
        )

        if self.CONTENT_TAG in response:
            response = self._iterate(response)

        return response

    def _post(
        self, endpoint: str, params: Dict[str, Any] = {}, data: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        request_params = {"api-version": self.api_version, "maxPageSize": 1000}
        request_params.update(params)
        response = self._request(
            "POST", endpoint, params=request_params, timeout=self.DEFAULT_TIMEOUT, data=data
        )

        if self.CONTENT_TAG in response:
            response = self._iterate(response)

        return response

    def get_seasonal_fields(self, party_id: str, params: Dict[str, Any] = {}):
        """Retrieves the seasonal fields for a given party.

        :param party_id: The ID of the party.

        :param params: Additional parameters to be passed to the request. Defaults to {}.

        :return: The information for each seasonal fields.
        """
        endpoint = f"/parties/{party_id}/seasonal-fields"
        request_params = {"api-version": self.api_version}
        request_params.update(params)

        return self._get(
            endpoint=endpoint,
            params=request_params,
        )

    def get_field(self, party_id: str, field_id: str):
        """
        Retrieves the field information for a given party and field.

        :param party_id: The ID of the party.

        :param field_id: The ID of the field.

        :return: The field information.
        """
        endpoint = f"/parties/{party_id}/fields/{field_id}"
        return self._get(endpoint)

    def get_seasonal_field(self, party_id: str, seasonal_field_id: str):
        """Retrieves the information of a seasonal field for a given party.

        :param party_id: The ID of the party.

        :param seasonal_field_id: The ID of the seasonal field.

        :return: The seasonal field information.
        """
        endpoint = f"/parties/{party_id}/seasonal-fields/{seasonal_field_id}"
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
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        operation_name: str,
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
        sources: List[str] = [],
    ):
        """
        Retrieves the information of a specified operation for a given party.

        This method will return information about the specified operation name,
        in the specified time range, for the given party and associated resource.

        :param party_id: The ID of the party.

        :param intersects_with_geometry: geometry of associated resource.

        :param operation_name: The name of the operation.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :param sources: (optional) The sources of the operation.

        :return: The operation information.
        """
        endpoint = f"/{operation_name}:search"
        params = {
            "api-version": self.api_version,
        }

        data = {
            "partyId": party_id,
            "intersectsWithGeometry": intersects_with_geometry,
            "minOperationStartDateTime": min_start_operation,
            "maxOperationEndDateTime": max_end_operation,
            "associatedResourceType": associated_resource["type"],
            "associatedResourceIds": [associated_resource["id"]],
        }

        if sources:
            data["sources"] = sources

        return self._post(endpoint, params=params, data=data)

    def get_harvest_info(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        """Retrieves the harvest information for a given party.

        This method will return the harvest information for a given resource,
        associated with the provided party id, between the start & end
        operation dates specified and intersecting with input geometry.

        :param party_id: ID of the party.

        :param intersects_with_geometry: geometry of associated resource.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with harvest information.
        """
        return self.get_operation_info(
            party_id=party_id,
            intersects_with_geometry=intersects_with_geometry,
            operation_name="harvest-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            associated_resource=associated_resource,
        )

    def get_fertilizer_info(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        """Retrieves the fertilizer information for a given party.

        This method will return the fertilizer information for a given resource,
        associated with the provided party id, between the start & end
        operation dates specified and intersecting with input geometry.

        :param party_id: ID of the party.

        :param intersects_with_geometry: geometry of associated resource.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with fertilizer information.
        """
        return self.get_operation_info(
            party_id=party_id,
            intersects_with_geometry=intersects_with_geometry,
            operation_name="application-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            sources=["Fertilizer"],
            associated_resource=associated_resource,
        )

    def get_organic_amendments_info(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        """Retrieves the organic amendments information for a given party.

        This method will return the organic amendments information for a given resource,
        associated with the provided party id, between the start & end
        operation dates specified and intersecting with input geometry.

        :param party_id: ID of the party.

        :param intersects_with_geometry: geometry of associated resource.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with organic amendments information.
        """

        return self.get_operation_info(
            party_id=party_id,
            intersects_with_geometry=intersects_with_geometry,
            operation_name="application-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            sources=["Omad"],
            associated_resource=associated_resource,
        )

    def get_tillage_info(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        """Retrieves the tillage information for a given party.

        This method will return the tillage information for a given resource,
        associated with the provided party id, between the start & end
        operation dates specified and intersecting with input geometry.

        :param party_id: ID of the Party.

        :param intersects_with_geometry: geometry of associated resource.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with tillage information.
        """
        return self.get_operation_info(
            party_id=party_id,
            intersects_with_geometry=intersects_with_geometry,
            operation_name="tillage-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            associated_resource=associated_resource,
        )

    def get_prescription_map_id(self, party_id: str, field_id: str, crop_id: str):
        """Retrieves the prescription map ID for a given party.

        This method will return the prescription map ID for a given party,
        associated with the provided field and crop IDs.

        :param party_id: ID of the Party.

        :param field_id: ID of the field.

        :param crop_id: ID of the crop.

        return: Dictionary with prescription map ID.
        """
        endpoint = f"parties/{party_id}/prescription-maps"
        return self._get(endpoint, params={"fieldIds": [field_id], "cropIds": [crop_id]})

    def get_prescriptions(
        self, party_id: str, prescription_map_id: str, geometry: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Retrieves the prescriptions for a given party.

        This method will return the prescriptions for a given party,
        associated with the provided prescription map ID.

        :param party_id: ID of the party.

        :param prescription_map_id: ID of the prescription map.

        :param geometry: geometry intersect with prescriptions.

        return: Dictionary with prescriptions.
        """
        endpoint = "/prescription:search"
        return self._post(
            endpoint,
            params={},
            data={
                "partyId": party_id,
                "prescriptionMapIds": [prescription_map_id],
                "intersectsWithGeometry": geometry,
            },
        )

    def get_prescription(self, party_id: str, prescription_id: str):
        """Retrieves the prescription for a given party.

        This method will return the prescription  for a given party,
        associated with the provided party_id.

        :param party_id: ID of the Party.

        :param prescription_id: ID of the prescription.

        return: Dictionary with prescription.
        """
        endpoint = f"parties/{party_id}/prescriptions/{prescription_id}"
        return self._get(endpoint)

    def get_planting_info(
        self,
        party_id: str,
        intersects_with_geometry: Dict[str, Any],
        min_start_operation: str,
        max_end_operation: str,
        associated_resource: Dict[str, str],
    ):
        """Retrieves the Planting information for a given resource.

        This method will return the Planting information for a given resource,
        associated with the provided party id, between the start & end
        operation dates specified and intersecting with input geometry.

        :param resource: resource linked to planting information.

        :param intersects_with_geometry: resource geometry.

        :param min_start_operation: The minimum start date of the operation.

        :param max_end_operation: The maximum end date of the operation.

        :return: Dictionary with planting information.
        """
        return self.get_operation_info(
            party_id=party_id,
            intersects_with_geometry=intersects_with_geometry,
            operation_name="planting-data",
            min_start_operation=min_start_operation,
            max_end_operation=max_end_operation,
            associated_resource=associated_resource,
        )
