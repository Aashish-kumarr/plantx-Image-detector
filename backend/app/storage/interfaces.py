"""Storage interface abstractions for binary image assets."""

from abc import ABC, abstractmethod
from typing import BinaryIO, Optional


class StorageInterface(ABC):
    """Abstract interface for image file storage (local disk / object storage)."""

    @abstractmethod
    def save(self, file_bytes: bytes, file_id: str, content_type: str) -> str:
        """Save raw bytes to storage and return storage key or URL."""
        pass

    @abstractmethod
    def get(self, storage_key: str) -> Optional[BinaryIO]:
        """Retrieve binary file stream by storage key."""
        pass

    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """Delete file from storage by storage key."""
        pass
