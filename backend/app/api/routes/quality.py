"""Quality check route — POST /api/v1/images/{image_id}/quality."""

from fastapi import APIRouter, Depends

from app.core.auth import AuthUser, get_current_user
from app.core.errors import ForbiddenError, ImageNotFoundError
from app.core.logging import get_logger
from app.schemas.quality import QualityResponse
from app.services import image_registry
from app.services.image_quality_service import image_quality_service

logger = get_logger("routes.quality")

router = APIRouter(tags=["Images"])


@router.post(
    "/images/{image_id}/quality",
    response_model=QualityResponse,
)
async def check_image_quality(
    image_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> QualityResponse:
    """Run the product image quality gate for a previously uploaded image.

    Returns a quality report indicating whether the image is suitable for
    plant disease analysis.

    NOTE: Sprint A returns a deterministic development stub result.
    Real blur/lighting/resolution algorithms are not yet implemented.
    """
    meta = image_registry.get(image_id)
    if meta is None:
        raise ImageNotFoundError(image_id)

    if meta.user_id and current_user.uid and meta.user_id != current_user.uid:
        raise ForbiddenError("You do not own this image.")

    result = image_quality_service.run_quality_check(meta)
    logger.info("Quality check for image_id=%s -> status=%s", image_id, result.status)
    return result

