"""Base model for trust runtime data classes."""

from pydantic import BaseModel, ConfigDict


class TrustModel(BaseModel):
    """Base model — extra fields ignored, JSON-mode serialization."""

    model_config = ConfigDict(extra="ignore")
