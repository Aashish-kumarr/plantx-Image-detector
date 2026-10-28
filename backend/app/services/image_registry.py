"""Lightweight in-memory image metadata registry for Sprint A.

After an image is uploaded it must be referenceable by image_id before an
analysis document is created.  This registry bridges the gap without
introducing a second MongoDB collection in this sprint.

NOTE: This is a development-only in-process store.  If the server restarts,
registered images are lost.  A persistent images collection will replace
this in a later sprint when auth and multi-process deployment are added.
"""

import threading
from typing import Dict, Optional

from app.schemas.image import ImageMeta

_lock = threading.Lock()
_registry: Dict[str, ImageMeta] = {}


def register(meta: ImageMeta) -> None:
    """Store image metadata under its image_id."""
    with _lock:
        _registry[meta.image_id] = meta


def get(image_id: str) -> Optional[ImageMeta]:
    """Return ImageMeta for ``image_id``, or None if not found."""
    with _lock:
        return _registry.get(image_id)


def remove(image_id: str) -> None:
    """Remove an image registration (e.g. after successful analysis)."""
    with _lock:
        _registry.pop(image_id, None)


def clear() -> None:
    """Clear all registrations — used only in tests."""
    with _lock:
        _registry.clear()
