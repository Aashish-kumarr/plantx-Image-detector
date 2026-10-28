"""Local filesystem storage adapter implementing StorageInterface.

Stores uploaded images under the configured LOCAL_STORAGE_ROOT directory.
Storage keys are UUID-based; original filenames are never used as paths.
"""

import os
import uuid
from typing import BinaryIO, Optional

from app.core.logging import get_logger
from app.storage.interfaces import StorageInterface

logger = get_logger("storage.local")


class LocalStorage(StorageInterface):
    """Development local disk storage adapter.

    Files are stored under ``root_dir`` using UUID-based filenames.
    Path-traversal protection is enforced on every operation.
    """

    def __init__(self, root_dir: str) -> None:
        self._root = os.path.realpath(root_dir)
        os.makedirs(self._root, exist_ok=True)
        logger.info("LocalStorage initialized at: %s", self._root)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _safe_path(self, storage_key: str) -> str:
        """Resolve ``storage_key`` relative to root and verify no traversal.

        Raises:
            ValueError: If the resolved path escapes the storage root.
        """
        # storage_key must be a bare filename (no path separators)
        # We join and then realpath to normalize any sneaky sequences.
        candidate = os.path.realpath(os.path.join(self._root, storage_key))
        if not candidate.startswith(self._root + os.sep) and candidate != self._root:
            raise ValueError(
                f"Storage key '{storage_key}' resolves outside the storage root."
            )
        return candidate

    @staticmethod
    def _generate_key(suffix: str = "") -> str:
        """Generate a UUID4-based storage key with an optional extension suffix."""
        name = uuid.uuid4().hex
        if suffix:
            # Ensure suffix starts with a dot and contains no path separators.
            suffix = suffix.lstrip(".")
            suffix = suffix.replace("/", "").replace("\\", "").replace("..", "")
            name = f"{name}.{suffix}"
        return name

    # ── StorageInterface implementation ───────────────────────────────────────

    def save(self, file_bytes: bytes, file_id: str, content_type: str) -> str:
        """Save ``file_bytes`` and return the generated storage key.

        Args:
            file_bytes: Raw image bytes to persist.
            file_id: Ignored — storage key is always generated internally.
            content_type: MIME type used to derive the file extension.

        Returns:
            The storage key (filename) of the saved file.
        """
        ext = _mime_to_ext(content_type)
        key = self._generate_key(ext)
        dest = self._safe_path(key)

        try:
            with open(dest, "wb") as fh:
                fh.write(file_bytes)
            logger.debug("Saved %d bytes -> %s", len(file_bytes), key)
            return key
        except OSError as exc:
            logger.error("Failed to write file %s: %s", key, exc)
            raise

    def get(self, storage_key: str) -> Optional[BinaryIO]:
        """Open ``storage_key`` for reading.

        Returns None if the file does not exist.
        """
        try:
            path = self._safe_path(storage_key)
        except ValueError:
            return None

        if not os.path.isfile(path):
            return None

        return open(path, "rb")  # noqa: SIM115 — caller is responsible for closing

    def read_bytes(self, storage_key: str) -> Optional[bytes]:
        """Read and return all bytes for ``storage_key``, or None if missing."""
        fh = self.get(storage_key)
        if fh is None:
            return None
        try:
            return fh.read()
        finally:
            fh.close()

    def delete(self, storage_key: str) -> bool:
        """Delete the file identified by ``storage_key``.

        Returns:
            True if deleted, False if the file did not exist.
        """
        try:
            path = self._safe_path(storage_key)
        except ValueError:
            return False

        if not os.path.isfile(path):
            return False

        try:
            os.remove(path)
            logger.debug("Deleted storage key: %s", storage_key)
            return True
        except OSError as exc:
            logger.error("Failed to delete %s: %s", storage_key, exc)
            return False

    def exists(self, storage_key: str) -> bool:
        """Return True if the storage key exists on disk."""
        try:
            path = self._safe_path(storage_key)
            return os.path.isfile(path)
        except ValueError:
            return False

    @property
    def root(self) -> str:
        """Absolute path to the storage root directory."""
        return self._root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mime_to_ext(mime_type: str) -> str:
    """Map a MIME type to a safe file extension."""
    mapping = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }
    return mapping.get(mime_type.lower(), "bin")
