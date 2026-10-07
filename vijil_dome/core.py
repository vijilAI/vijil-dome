"""VijilDome — policy-driven runtime for agent controls.

The policy is the single config surface.  All detection — ML models,
regex, custom evaluators — is defined in the policy controls.  Dome
detectors are referenced via ``dome:*`` evaluators.

The existing :class:`~vijil_dome.Dome.Dome` class is unchanged and
continues to work for standalone content guardrailing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

from vijil_dome.controls.decorator import control as control_decorator
from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.errors import handle_result
from vijil_dome.controls.models import Control, EvaluationResult, Step
from vijil_dome.utils.api_config_loader import DEFAULT_CONSOLE_API_BASE_URL

logger = logging.getLogger(__name__)


class VijilDome:
    """Policy-driven runtime.

    All detection — ML models, regex, custom evaluators — is defined
    in the policy controls.  Dome detectors are referenced via
    ``dome:*`` evaluators within the policy.

    Parameters
    ----------
    policy:
        Controls definition.  Can be:
        - A list of :class:`Control` objects or dicts
        - A path (str or Path) to a YAML/JSON file
        - ``None`` (empty policy — controls can be added later)
    enforce:
        If ``True`` (default), deny/steer actions raise exceptions.
        If ``False``, violations are logged but not enforced (shadow mode).
    agent_id:
        Optional agent identifier for audit/logging.
    team_id:
        Optional team identifier for audit/logging and multi-tenant
        telemetry filtering (e.g. the ``team.id`` resource attribute
        Console's trace search already relies on).
    user_id:
        Optional end-user identifier for audit/logging.
    """

    def __init__(
        self,
        *,
        policy: list[dict | Control] | str | Path | None = None,
        enforce: bool = True,
        agent_id: str | None = None,
        team_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._enforce = enforce
        self._agent_id = agent_id
        self._team_id = team_id
        self._user_id = user_id
        self._engine = ControlEngine()

        # Populated by create_from_s3(); let config_has_changed() detect an
        # instance that wasn't created that way, mirroring Dome.create_from_s3.
        self._s3_bucket: str | None = None
        self._s3_key: str | None = None
        self._s3_config_dict: dict[str, Any] | None = None
        self._s3_aws_kwargs: dict[str, Any] | None = None
        self._s3_cache_dir: str | None = None

        # Populated by create_from_api(); see config_has_changed().
        self._api_base_url: str | None = None
        self._api_client_id: str | None = None
        self._api_client_secret: str | None = None
        self._api_agent_id: str | None = None
        self._api_team_id: str | None = None
        self._api_config_dict: dict[str, Any] | None = None
        self._api_cache_dir: str | None = None
        self._api_access_token: str | None = None
        self._api_token_expires_at: float | None = None

        if isinstance(policy, (str, Path)):
            self._engine.load_controls_from_file(str(policy))
        elif isinstance(policy, list):
            defs = [
                c if isinstance(c, dict) else c.model_dump(by_alias=True)
                for c in policy
            ]
            self._engine.load_controls(defs)

    @staticmethod
    def create_from_s3(
        bucket: str,
        key: str | None = None,
        team_id: str | None = None,
        agent_id: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str | None = None,
        cache_dir: str | None = None,
        cache_ttl_seconds: int = 300,
        enforce: bool = True,
    ) -> "VijilDome":
        """Create a VijilDome instance from a config stored in S3.

        Reads the same ``config.json`` Console's agent-scoped ``DomeConfig``
        already writes (and the old :class:`~vijil_dome.Dome.Dome` class
        already reads for its legacy ``input-guards``/``output-guards``
        format) — this just also reads that document's ``controls`` key.
        TTL/ETag caching is unchanged, reused as-is from
        :func:`~vijil_dome.utils.config_loader.load_dome_config_from_s3`; see
        ``docs/policy-enforcement/2026-10-06-policy-enforcement-scoping.md``
        §3.6/§5B for why no new freshness mechanism was needed here.

        The S3 key can be provided directly, or constructed from *team_id*
        and *agent_id* using the standard path
        ``teams/{team_id}/agents/{agent_id}/dome/config.json``.

        The loaded config is cached locally and the S3 coordinates are
        stored on the instance so that :meth:`config_has_changed` can later
        check for remote updates.
        """
        from vijil_dome.utils.config_loader import (
            _resolve_key,
            load_dome_config_from_s3,
        )

        aws_kwargs = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "region_name": region_name,
        }
        config_dict = load_dome_config_from_s3(
            bucket=bucket,
            key=key,
            team_id=team_id,
            agent_id=agent_id,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            **aws_kwargs,
        )

        controls_list = config_dict.get("controls", [])
        vijil_dome = VijilDome(
            policy=controls_list, enforce=enforce, agent_id=agent_id, team_id=team_id
        )

        vijil_dome._s3_bucket = bucket
        vijil_dome._s3_key = _resolve_key(key, team_id, agent_id)
        vijil_dome._s3_config_dict = config_dict
        vijil_dome._s3_aws_kwargs = aws_kwargs
        vijil_dome._s3_cache_dir = cache_dir
        return vijil_dome

    @staticmethod
    def create_from_api(
        client_id: str,
        client_secret: str,
        agent_id: str,
        team_id: str,
        base_url: str = DEFAULT_CONSOLE_API_BASE_URL,
        cache_dir: str | None = None,
        cache_ttl_seconds: int = 300,
        enforce: bool = True,
    ) -> "VijilDome":
        """Create a VijilDome instance from Console's API, instead of S3.

        No S3 bucket name or AWS credentials needed. *client_id* and
        *client_secret* are a Console API key -- mint one for the team on
        Console's API Keys page (Profile > API Keys) -- exchanged here for a
        short-lived JWT, which is then used to call Console's
        ``GET /{agent_id}/dome-configs/active`` endpoint. That endpoint only
        ever returns a config with ``status = active``, so this can't load
        an unreviewed pending draft.

        The loaded config is cached locally (TTL ``cache_ttl_seconds``) and
        the API coordinates are stored on the instance so that
        :meth:`config_has_changed` can later check for remote updates.
        """
        from vijil_dome.utils.api_config_loader import load_dome_config_from_api

        config_dict, access_token, token_expires_at = load_dome_config_from_api(
            base_url=base_url,
            client_id=client_id,
            client_secret=client_secret,
            agent_id=agent_id,
            team_id=team_id,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        controls_list = config_dict.get("controls", [])
        vijil_dome = VijilDome(
            policy=controls_list, enforce=enforce, agent_id=agent_id, team_id=team_id
        )

        vijil_dome._api_base_url = base_url
        vijil_dome._api_client_id = client_id
        vijil_dome._api_client_secret = client_secret
        vijil_dome._api_agent_id = agent_id
        vijil_dome._api_team_id = team_id
        vijil_dome._api_config_dict = config_dict
        vijil_dome._api_cache_dir = cache_dir
        vijil_dome._api_access_token = access_token
        vijil_dome._api_token_expires_at = token_expires_at
        return vijil_dome

    def config_has_changed(self) -> bool:
        """Check whether the remote config has changed since this instance was created.

        Only works for instances created via :meth:`create_from_s3` or
        :meth:`create_from_api`.

        Returns:
            ``True`` if the remote config differs from the one used to
            create this instance, ``False`` otherwise.

        Raises:
            ValueError: If the instance was not created from S3 or the API.
        """
        if self._s3_bucket is not None and self._s3_key is not None:
            from vijil_dome.utils.config_loader import config_has_changed as _config_has_changed

            return _config_has_changed(
                local_config=self._s3_config_dict,  # type: ignore[arg-type]
                bucket=self._s3_bucket,
                key=self._s3_key,
                cache_dir=self._s3_cache_dir,
                **(self._s3_aws_kwargs or {}),
            )

        if self._api_base_url is not None and self._api_agent_id is not None:
            from vijil_dome.utils.api_config_loader import api_config_has_changed

            return api_config_has_changed(
                local_config=self._api_config_dict,  # type: ignore[arg-type]
                base_url=self._api_base_url,
                client_id=self._api_client_id,  # type: ignore[arg-type]
                client_secret=self._api_client_secret,  # type: ignore[arg-type]
                agent_id=self._api_agent_id,
                team_id=self._api_team_id,  # type: ignore[arg-type]
                cache_dir=self._api_cache_dir,
            )

        raise ValueError(
            "config_has_changed() is only available for VijilDome instances "
            "created via VijilDome.create_from_s3() or VijilDome.create_from_api()."
        )

    @property
    def engine(self) -> ControlEngine:
        """The underlying control engine."""
        return self._engine

    @property
    def agent_id(self) -> str | None:
        return self._agent_id

    @property
    def team_id(self) -> str | None:
        return self._team_id

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def enforce(self) -> bool:
        return self._enforce

    # ------------------------------------------------------------------
    # Guard methods
    # ------------------------------------------------------------------

    def guard_input(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "input",
    ) -> EvaluationResult:
        """Evaluate input against the policy (sync)."""
        step = Step(
            type="llm",
            name=step_name,
            input=text,
            context=context or {},
        )
        result = self._engine.evaluate_sync(step, stage="pre")
        handle_result(result, self._enforce, "pre")
        return result

    async def async_guard_input(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "input",
    ) -> EvaluationResult:
        """Evaluate input against the policy (async)."""
        step = Step(
            type="llm",
            name=step_name,
            input=text,
            context=context or {},
        )
        result = await self._engine.evaluate(step, stage="pre")
        handle_result(result, self._enforce, "pre")
        return result

    def guard_output(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "output",
    ) -> EvaluationResult:
        """Evaluate output against the policy (sync)."""
        step = Step(
            type="llm",
            name=step_name,
            output=text,
            context=context or {},
        )
        result = self._engine.evaluate_sync(step, stage="post")
        handle_result(result, self._enforce, "post")
        return result

    async def async_guard_output(
        self,
        text: str | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        step_name: str = "output",
    ) -> EvaluationResult:
        """Evaluate output against the policy (async)."""
        step = Step(
            type="llm",
            name=step_name,
            output=text,
            context=context or {},
        )
        result = await self._engine.evaluate(step, stage="post")
        handle_result(result, self._enforce, "post")
        return result

    def guard_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate a tool call against the policy (sync)."""
        step = Step(
            type="tool",
            name=tool_name,
            input=tool_input,
            context=context or {},
        )
        result = self._engine.evaluate_sync(step, stage="pre")
        handle_result(result, self._enforce, "pre")
        return result

    async def async_guard_tool_call(
        self,
        tool_name: str,
        tool_input: Any = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate a tool call against the policy (async)."""
        step = Step(
            type="tool",
            name=tool_name,
            input=tool_input,
            context=context or {},
        )
        result = await self._engine.evaluate(step, stage="pre")
        handle_result(result, self._enforce, "pre")
        return result

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def control(
        self,
        *,
        step_type: Literal["tool", "llm"] = "llm",
        step_name: str | None = None,
        input_mapper: Callable | None = None,
        output_mapper: Callable | None = None,
        context_mapper: Callable | None = None,
    ) -> Callable:
        """Return a decorator that uses this instance's engine and enforce mode."""
        return control_decorator(
            engine=self._engine,
            step_type=step_type,
            step_name=step_name,
            enforce=self._enforce,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
            context_mapper=context_mapper,
        )

