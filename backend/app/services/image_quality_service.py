"""Product image quality service — Sprint A deterministic development stub.

IMPORTANT — TEMPORARY IMPLEMENTATION:
    This service does NOT implement real blur detection, resolution measurement,
    lighting analysis, or contrast computation.

    It returns a deterministic "all checks passed" response for every valid
    image to unblock end-to-end API integration during Sprint A.

    Real quality algorithms will be implemented in a later sprint once
    field-validation images and concrete thresholds are established
    (see docs/backend-architecture.md, docs/api-contract.md Section 2).

Responsibility boundary:
    This service is a PRODUCT quality gate ("Is this photograph suitable for
    analysis?"), NOT an ML/DSP component.  It lives under app/services/,
    not app/ml/.
"""

from app.core.logging import get_logger
from app.schemas.image import ImageMeta
from app.schemas.quality import QualityCheck, QualityChecks, QualityResponse

logger = get_logger("services.image_quality")


class ImageQualityService:
    """Evaluate whether an uploaded image is suitable for plant disease analysis.

    Sprint A implementation: deterministic development stub.
    All checks return "good" for any image that passed upload validation.
    """

    def run_quality_check(self, image_meta: ImageMeta) -> QualityResponse:
        """Return a deterministic quality result for the provided image metadata.

        Args:
            image_meta: Metadata for the previously uploaded and validated image.

        Returns:
            QualityResponse with all checks marked "good" and overall status
            "passed".  This is a development fixture only.
        """
        logger.debug(
            "Quality check (stub) for image_id=%s — returning deterministic passed result.",
            image_meta.image_id,
        )

        good = QualityCheck(status="good", reason=None)

        return QualityResponse(
            status="passed",
            checks=QualityChecks(
                resolution=good,
                sharpness=good,
                lighting=good,
                contrast=good,
            ),
        )


# Module-level singleton for injection / testing convenience.
image_quality_service = ImageQualityService()
