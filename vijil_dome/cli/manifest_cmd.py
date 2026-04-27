"""vijil manifest sign|verify — CLI commands for tool manifest management."""

import json
import sys
from pathlib import Path

import typer

from vijil_dome.trust.manifest import ToolManifest


def register_manifest(app: typer.Typer) -> None:
    """Register the manifest sub-commands on the main app."""
    manifest_app = typer.Typer(
        name="manifest",
        help="Sign and verify tool manifests.",
        no_args_is_help=True,
    )
    app.add_typer(manifest_app)

    @manifest_app.command("sign")
    def sign(
        input_path: str = typer.Argument(help="Path to unsigned manifest JSON"),
        console_url: str = typer.Option(
            ..., "--console-url", envvar="VIJIL_CONSOLE_URL",
            help="Vijil Console base URL",
        ),
        api_key: str = typer.Option(
            ..., "--api-key", envvar="VIJIL_API_KEY",
            help="Vijil Console API key",
        ),
        output: str | None = typer.Option(
            None, "--output", "-o", help="Output path (defaults to overwriting input)"
        ),
    ) -> None:
        """Sign a tool manifest via the Vijil Console."""
        import httpx

        input_file = Path(input_path)
        manifest_data: dict[str, object] = json.loads(input_file.read_text())

        resp = httpx.post(
            f"{console_url.rstrip('/')}/manifests/sign",
            json=manifest_data,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        manifest_data["signature"] = resp.json()["signature"]

        dest = Path(output) if output else input_file
        dest.write_text(json.dumps(manifest_data, indent=2))
        typer.echo(f"Signed manifest written to {dest}")

    @manifest_app.command("verify")
    def verify(
        path: str = typer.Argument(help="Path to signed manifest JSON"),
        console_url: str = typer.Option(
            ..., "--console-url", envvar="VIJIL_CONSOLE_URL",
            help="Vijil Console base URL",
        ),
        api_key: str = typer.Option(
            ..., "--api-key", envvar="VIJIL_API_KEY",
            help="Vijil Console API key",
        ),
    ) -> None:
        """Verify a tool manifest's signature against the Vijil Console public key."""
        import httpx
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        manifest = ToolManifest.load(Path(path))

        resp = httpx.get(
            f"{console_url.rstrip('/')}/manifests/public-key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        pem_bytes = resp.json()["public_key"].encode()
        public_key = load_pem_public_key(pem_bytes)

        if manifest.verify_signature(public_key):  # type: ignore[arg-type]
            typer.echo("Manifest signature valid.")
        else:
            typer.echo("Manifest signature INVALID.", err=True)
            sys.exit(4)
