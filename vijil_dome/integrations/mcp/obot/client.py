import requests
from typing import Optional, Dict, Any, List


SUPPORTED_LOCAL_COMMANDS = ["npx", "uvx"]


def extract_npx_package(args):
    """
    Given a list of npx arguments, return (package_name, remaining_args).

    Example:
    ['-y', '@scope/pkg', 'init'] → ('@scope/pkg', ['init'])
    """
    skip_flags = {"-y", "--yes"}

    if len(args) == 0:
        return "", []

    # Remove leading flags
    filtered = []
    for a in args:
        if a in skip_flags:
            continue
        filtered.append(a)

    if not filtered:
        return None, []

    package_name = filtered[0]
    remaining = filtered[1:]

    return package_name, remaining


def extract_uvx_package(args):
    """
    Given a list of uvx arguments, return (package_name, remaining_args).

    uvx syntax is:
      uvx [flags...] <package-or-script> [args...]

    Example:
    ['--python', '3.12', 'ruff', 'check'] → ('ruff', ['check'])
    """

    # Flags for uvx generally start with -
    filtered = []
    i = 0

    if len(args) == 0:
        return "", []

    while i < len(args):
        if args[i].startswith("-"):
            # Skip flag and its value if it expects one (we’ll ignore exhaustive handling)
            filtered.append(None)  # placeholder for skipped flags
            i += 1
            # If next arg does NOT start with '-', assume it's a value for this flag
            if i < len(args) and not args[i].startswith("-"):
                i += 1
        else:
            # First non-flag is the package
            break

    if i >= len(args):
        return None, []

    package_name = args[i]
    remaining = args[i + 1 :]

    return package_name, remaining


def format_kv_pair(dict: Dict[str, Any]) -> List[dict]:
    values = []
    for var in dict:
        data_object = {
            "key": var,
            "value": dict[var],
            "description": "",
            "file": False,
            "sensitive": False,
            "name": "",
        }
        values.append(data_object)
    return values


class ObotClient:
    """
    A lightweight OBOT API client that:
      - Obtains an API token automatically if needed
      - Creates an MCP Server (multi-user server)
      - Returns the connect URL for that server
    """

    def __init__(self, obot_url: str, token: Optional[str] = None):
        """
        :param obot_url: The base URL of the OBOT instance, e.g. "http://localhost:8080"
        :param token: Optional bearer token. If not provided, the client fetches one.
        """
        self.obot_url = obot_url.rstrip("/")
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _create_local_manifest(
        self, server_name: str, local_mcp_server_config: dict
    ) -> Dict[str, Any]:
        manifest = {
            "name": server_name,
            "description": "",  # Not part of a normal MCP config
            "icon": "",  # not part of normal mcp
            "metadata": {"categories": ""},  # not part of normal mcp
        }  # type: Dict[str, Any]

        env_vars = format_kv_pair(local_mcp_server_config.get("env", []))
        manifest["env"] = env_vars

        command = local_mcp_server_config.get("command")
        if command not in SUPPORTED_LOCAL_COMMANDS:
            raise ValueError(
                f"Unsupport command: {command}. command must be one of {SUPPORTED_LOCAL_COMMANDS}"
            )

        manifest["runtime"] = command
        runtime_config_key = f"{command}Config"

        if command == "npx":
            package, args = extract_npx_package(local_mcp_server_config.get("args", []))
        else:
            package, args = extract_uvx_package(local_mcp_server_config.get("args", []))
            print(package, args)

        args = format_kv_pair(args)

        runtime_config = {"package": package, "args": args}
        manifest[runtime_config_key] = runtime_config
        return manifest

    def _create_remote_manifest(
        self, server_name: str, remote_mcp_server_config: dict
    ) -> Dict[str, Any]:
        manifest = {
            "name": server_name,
            "description": "",  # Not part of a normal MCP config
            "icon": "",  # not part of normal mcp
            "metadata": {"categories": ""},  # not part of normal mcp
        }
        headers = format_kv_pair(remote_mcp_server_config.get("headers", []))
        manifest["runtime"] = "remote"
        runtime_config = {
            "url": remote_mcp_server_config.get("url"),
            "headers": headers,
        }
        manifest["remoteConfig"] = runtime_config
        return manifest

    def _create_mcp_server(
        self, server_name: str, mcp_server_config: dict
    ) -> Dict[str, Any]:
        """
        Creates an MCP server using the manifest dict.
        Returns the full server object returned by OBOT.
        """
        url = f"{self.obot_url}/api/mcp-catalogs/default/servers"

        is_cmd_based = "command" in mcp_server_config
        is_url_based = "url" in mcp_server_config
        if not is_cmd_based and not is_url_based:
            raise ValueError(
                "Atleast one of 'command' or 'url' must be present in the MCP server configuration"
            )
        if is_cmd_based and is_url_based:
            raise ValueError(
                "Both 'command' and 'url' components cannot be within the same server configuration"
            )

        if is_cmd_based:
            manifest = self._create_local_manifest(server_name, mcp_server_config)
        else:
            manifest = self._create_remote_manifest(server_name, mcp_server_config)

        response = requests.post(
            url, json={"manifest": manifest}, headers=self._headers()
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to create MCP server: {response.status_code} {response.text}"
            )

        return response.json()

    def get_connect_url(self, server_id: str) -> str:
        """
        Given a server ID, return the connect URL.
        """
        return f"{self.obot_url}/mcp-connect/{server_id}"

    def create_server_and_get_connect_url(
        self, server_name: str, server_config: Dict[str, Any]
    ) -> str:
        """
        Creates an MCP server and returns the connect URL for a single server.
        """
        server_obj = self._create_mcp_server(server_name, server_config)
        server_id = server_obj.get("id")
        if not server_id:
            raise RuntimeError("OBOT did not return a server ID.")

        return self.get_connect_url(server_id)

    def create_secure_mcp_config(self, mcp_config: dict) -> dict:
        """
        Creates a secure gateway with all the servers in the MCP Config
        Returns a new MCP config
        """
        if "mcpServers" not in mcp_config:
            raise ValueError("The first, and only key in the config must be mcpServers")
        secure_config = {}  # type: Dict[str, Any]
        for server in mcp_config["mcpServers"]:
            connect_url = self.create_server_and_get_connect_url(
                server, mcp_config["mcpServers"][server]
            )
            secure_config[server] = {}
            secure_config[server]["url"] = connect_url
            secure_config[server]["transport"] = "streamable_http"

        return {"mcpServers": secure_config}
