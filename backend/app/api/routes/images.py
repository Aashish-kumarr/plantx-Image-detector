"""Image upload route — POST /api/v1/images.

Performs authoritative image validation using Pillow, normalizes EXIF
orientation, and stores the file via LocalStorage.
"""

import io
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.auth import AuthUser, get_current_user
from app.core.config import settings

from app.core.errors import (
    InvalidImageError,
    ImageTooLargeError,
    StorageError,
    UnsupportedImageTypeError,
)
from app.core.logging import get_logger
from app.schemas.image import ImageMeta, ImageUploadResponse
from app.services import image_registry
from app.storage.local import LocalStorage

logger = get_logger("routes.images")

router = APIRouter(tags=["Images"])

# Allowed Pillow format identifiers after decode.
_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

# Singleton storage instance.
_storage: LocalStorage | None = None


def _get_storage() -> LocalStorage:
    """Return the module-level LocalStorage singleton, creating it if needed."""
    global _storage
    if _storage is None:
        _storage = LocalStorage(settings.LOCAL_STORAGE_ROOT)
    return _storage


@router.post("/images", response_model=ImageUploadResponse, status_code=201)
async def upload_image(
    image: Annotated[UploadFile, File(description="Plant leaf image (JPEG, PNG, or WebP, max 10 MB)")],
    current_user: AuthUser = Depends(get_current_user),
) -> ImageUploadResponse:

    """Upload and validate a plant leaf image.

    Validation steps (all authoritative — not based on extension or Content-Type alone):

    1. File must not be empty.
    2. Total size must not exceed MAX_UPLOAD_BYTES (10 MB).
    3. Pillow must successfully decode the file.
    4. Decoded format must be JPEG, PNG, or WebP.
    5. EXIF orientation is normalized before storage.
    6. Width and height are extracted from the decoded image.

    Returns:
        Image metadata including a server-generated image_id.
    """
    # ── Read all bytes from the upload ────────────────────────────────────────
    raw_bytes = await image.read()

    if not raw_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    if len(raw_bytes) > settings.MAX_UPLOAD_BYTES:
        max_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ImageTooLargeError(max_mb)

    # ── Authoritative format validation via Pillow ────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.verify()  # Checks file integrity without loading pixel data.
    except UnidentifiedImageError:
        raise InvalidImageError("The uploaded file is not a recognized image format.")
    except Exception:  # noqa: BLE001
        raise InvalidImageError("The uploaded file is corrupted or cannot be decoded.")

    # Re-open after verify() (Pillow resets the stream after verify).
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        # Load pixel data to catch truncated files.
        pil_image.load()
    except Exception:  # noqa: BLE001
        raise InvalidImageError("The uploaded file is corrupted or truncated.")

    pil_format = pil_image.format  # "JPEG", "PNG", "WEBP", etc.

    if pil_format not in _ALLOWED_FORMATS:
        raise UnsupportedImageTypeError()

    # ── EXIF orientation normalization ────────────────────────────────────────
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:  # noqa: BLE001
        # exif_transpose can fail on some malformed tags; proceed without it.
        logger.warning("EXIF transpose failed for upload — proceeding without normalization.")

    width, height = pil_image.size

    # ── Re-encode normalized image ────────────────────────────────────────────
    buf = io.BytesIO()
    save_format = pil_format if pil_format != "JPEG" else "JPEG"
    pil_image.save(buf, format=save_format)
    normalized_bytes = buf.getvalue()

    # ── Store file ────────────────────────────────────────────────────────────
    mime_type = _FORMAT_TO_MIME[pil_format]
    image_id = uuid.uuid4().hex

    try:
        storage = _get_storage()
        storage_key = storage.save(
            file_bytes=normalized_bytes,
            file_id=image_id,
            content_type=mime_type,
        )
    except OSError as exc:
        logger.error("Storage write failed: %s", exc)
        raise StorageError()

    # ── Register metadata ─────────────────────────────────────────────────────
    original_filename = image.filename or "upload"
    meta = ImageMeta(
        image_id=image_id,
        storage_key=storage_key,
        filename=original_filename,
        mime_type=mime_type,
        size_bytes=len(normalized_bytes),
        width=width,
        height=height,
        user_id=current_user.uid,
    )

    image_registry.register(meta)

    logger.info(
        "Image uploaded: image_id=%s format=%s size=%d width=%d height=%d",
        image_id,
        pil_format,
        len(normalized_bytes),
        width,
        height,
    )

    return ImageUploadResponse(
        image_id=image_id,
        filename=original_filename,
        mime_type=mime_type,
        size_bytes=len(normalized_bytes),
        width=width,
        height=height,
    )
