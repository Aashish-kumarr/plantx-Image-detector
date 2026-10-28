"""Pydantic schemas for uploaded image metadata."""

from typing import Optional
from pydantic import BaseModel, Field



class ImageUploadResponse(BaseModel):
    """Response returned after a successful image upload."""

    image_id: str = Field(..., description="Unique image identifier (UUID)")
    filename: str = Field(..., description="Original uploaded filename")
    mime_type: str = Field(..., description="Detected MIME type (image/jpeg, image/png, image/webp)")
    size_bytes: int = Field(..., description="File size in bytes after normalization")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class ImageMeta(BaseModel):
    """Internal image metadata record stored alongside an analysis."""

    image_id: str
    storage_key: str
    filename: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    user_id: Optional[str] = None

