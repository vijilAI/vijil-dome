"""vijil manifest sign|verify"""

import json
import sys
from pathlib import Path

import typer

from vijil_dome.trust.manifest import ToolManifest
# TODO: rewire for vijil-dome CLI — vijil_cli.state is SDK-specific
# from vijil_cli import state


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
        output: str | None = typer.Option(
            None, "--output", "-o", help="Output path (defaults to overwriting input)"
        ),
    ) -> None:
        """Sign a tool manifest via the Vijil Console."""
        ctx = state.get_ctx()
        input_file = Path(input_path)
        manifest_data: dict[str, object] = json.loads(input_file.read_text())

        result = ctx.client._http.post("/manifests/sign", json=manifest_data)
        manifest_data["signature"] = result["signature"]

        dest = Path(output) if output else input_file
        dest.write_text(json.dumps(manifest_data, indent=2))
        typer.echo(f"Signed manifest written to {dest}")

    @manifest_app.command("verify")
    def verify(
        path: str = typer.Argument(help="Path to signed manifest JSON"),
    ) -> None:
        """Verify a tool manifest's signature against the Vijil Console public key."""
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        ctx = state.get_ctx()
        manifest = ToolManifest.load(Path(path))

        result = ctx.client._http.get("/manifests/public-key")
        pem_bytes = result["public_key"].encode()
        public_key = load_pem_public_key(pem_bytes)

        if manifest.verify_signature(public_key):  # type: ignore[arg-type]
            typer.echo("Manifest signature valid.")
        else:
            typer.echo("Manifest signature INVALID.", err=True)
            sys.exit(4)
