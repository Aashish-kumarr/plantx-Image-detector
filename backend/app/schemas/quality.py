"""Pydantic schemas for image quality check results.

Matches docs/api-contract.md — Section 2: Image Quality Check.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class QualityCheck(BaseModel):
    """Result of a single quality dimension check."""

    status: Literal["good", "needs_improvement"] = Field(
        ..., description="Semantic check outcome"
    )
    reason: Optional[str] = Field(
        None,
        description=(
            "Machine-readable reason code when status is needs_improvement. "
            "Possible values: low_resolution, blur_detected, underexposed, "
            "overexposed, low_contrast."
        ),
    )


class QualityChecks(BaseModel):
    """Container for all four product image quality dimensions."""

    resolution: QualityCheck
    sharpness: QualityCheck
    lighting: QualityCheck
    contrast: QualityCheck


class QualityResponse(BaseModel):
    """Response for POST /api/v1/images/{image_id}/quality."""

    status: Literal["passed", "needs_improvement"] = Field(
        ..., description="Overall quality gate outcome"
    )
    checks: QualityChecks
