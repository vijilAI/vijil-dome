"""
Integrations with Vijil Evaluate
"""

import httpx
from typing import Optional, Dict, Any

VIJIL_API_BASE_URL = "https://evaluate-api.vijil.ai/v1"


def get_config_from_vijil_agent(
    api_token: str, agent_id: str, base_url: Optional[str] = None
) -> Optional[dict]:
    """
    Fetch the Dome configuration from Vijil Evaluate using the provided API token and agent ID.

    Args:
        api_token (str): The API token for authentication.
        agent_id (str): The ID of the agent whose configuration is to be fetched.

    Returns:
        dict: The Dome configuration as a dictionary.

    Raises:
        Exception: If the API call fails or returns an error.
    """
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    base_url = base_url or VIJIL_API_BASE_URL
    url = f"{base_url}/agent-configurations/{agent_id}/dome-configs"

    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        dome_configs = response_json.get("dome_configs")
        if dome_configs is None:
            raise Exception("Dome configuration not found in the response.")
        else:
            if len(dome_configs) == 0:
                return None
            else:
                config = dome_configs[0].get("config_body", None)
                return config
    except httpx.HTTPError as e:
        raise Exception(f"Failed to fetch Dome config: {e}")


def get_config_from_vijil_evaluation(
    api_token: str,
    evaluation_id: str,
    base_url: Optional[str] = None,
    latency_threshold: Optional[float] = None,
) -> Optional[dict]:
    """
    Fetch the Dome configuration from a specific evaluation in Vijil Evaluate using the provided API token and evaluation ID.

    Args:
        api_token (str): The API token for authentication.
        evaluation_id (str): The ID of the evaluation whose configuration is to be fetched.

    Returns:
        dict: The Dome configuration as a dictionary.

    Raises:
        Exception: If the API call fails or returns an error.
    """
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    base_url = base_url or VIJIL_API_BASE_URL
    url = f"{base_url}/recommend-dome-config"

    payload = {"evaluation_id": evaluation_id} # type: Dict[str, Any]
    if latency_threshold:
        payload["latency_threshold"] = latency_threshold

    try:
        response = httpx.post(url, headers=headers, json=payload)
        response.raise_for_status()
        dome_config = response.json()
        if dome_config is None:
            raise Exception("Dome configuration not found in the response.")
        else:
            return dome_config
    except httpx.HTTPError as e:
        raise Exception(f"Failed to fetch Dome config: {e}")
