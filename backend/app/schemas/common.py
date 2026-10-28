"""Shared common schemas for API responses and errors."""

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Structured machine error detail object."""

    code: str = Field(..., description="Machine-readable error code string")
    message: str = Field(..., description="User-safe error description")


class ErrorResponse(BaseModel):
    """Standard top-level error response envelope."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check status response schema."""

    status: str = Field("ok", description="Service status indicator")
    service: str = Field("PlantDx API", description="Service name")
    environment: str = Field("development", description="Current execution environment")
