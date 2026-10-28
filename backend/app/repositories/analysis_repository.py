"""Analysis repository — owns all MongoDB interactions for the analyses collection.

Routes and services must never query MongoDB directly.

Analysis IDs are UUID4 strings (field: ``id``).
MongoDB _id (ObjectId) is internal and never exposed in the API.
Timestamps use UTC-aware datetimes.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.core.errors import AnalysisNotFoundError, DatabaseError
from app.core.logging import get_logger
from app.db.client import get_database

logger = get_logger("repositories.analysis")

# Pagination guard
_MAX_LIMIT = 100


def _collection() -> Collection:
    """Return the analyses collection from the active database."""
    return get_database()["analyses"]


def _strip_mongo_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the internal MongoDB _id field before returning a document."""
    doc.pop("_id", None)
    return doc


class AnalysisRepository:
    """CRUD operations for the ``analyses`` MongoDB collection."""

    # ── Create ────────────────────────────────────────────────────────────────

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new analysis document.

        Args:
            data: Analysis data dict (without ``id`` or timestamps — these
                  are generated here).

        Returns:
            The created document dict (without MongoDB _id).
        """
        now = datetime.now(timezone.utc)
        doc: Dict[str, Any] = {
            **data,
            "id": uuid.uuid4().hex,
            "created_at": now,
            "updated_at": now,
        }
        try:
            _collection().insert_one(doc)
        except PyMongoError as exc:
            logger.error("Failed to insert analysis: %s", exc)
            raise DatabaseError() from exc

        return _strip_mongo_id(doc)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_id(self, analysis_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve a single analysis by its UUID string identifier.

        If user_id is provided, enforces that the analysis belongs to that user.

        Raises:
            AnalysisNotFoundError: If no document matches analysis_id or user ownership.
        """
        try:
            query: Dict[str, Any] = {"id": analysis_id}
            if user_id is not None:
                query["user_id"] = user_id
            doc = _collection().find_one(query)
        except PyMongoError as exc:
            logger.error("Failed to fetch analysis %s: %s", analysis_id, exc)
            raise DatabaseError() from exc

        if doc is None:
            raise AnalysisNotFoundError(analysis_id)

        return _strip_mongo_id(doc)

    def list(
        self,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Return a paginated list of analyses for user_id, ordered newest first.

        Args:
            user_id: Optional user identifier to filter by.
            skip: Number of documents to skip.
            limit: Maximum number of documents to return (capped at ``_MAX_LIMIT``).

        Returns:
            A tuple of (items, total_count).
        """
        limit = min(limit, _MAX_LIMIT)
        skip = max(skip, 0)
        query: Dict[str, Any] = {"user_id": user_id} if user_id is not None else {}

        try:
            col = _collection()
            total = col.count_documents(query)
            cursor = col.find(query).sort("created_at", -1).skip(skip).limit(limit)
            items = [_strip_mongo_id(doc) for doc in cursor]
        except PyMongoError as exc:
            logger.error("Failed to list analyses: %s", exc)
            raise DatabaseError() from exc

        return items, total


    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, analysis_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Apply partial updates to an existing analysis document.

        Args:
            analysis_id: UUID string of the document to update.
            updates: Fields to set (``updated_at`` is always refreshed).

        Returns:
            The updated document dict.

        Raises:
            AnalysisNotFoundError: If no document matches ``analysis_id``.
        """
        updates["updated_at"] = datetime.now(timezone.utc)
        try:
            result = _collection().find_one_and_update(
                {"id": analysis_id},
                {"$set": updates},
                return_document=True,
            )
        except PyMongoError as exc:
            logger.error("Failed to update analysis %s: %s", analysis_id, exc)
            raise DatabaseError() from exc

        if result is None:
            raise AnalysisNotFoundError(analysis_id)

        return _strip_mongo_id(result)


# Module-level singleton.
analysis_repository = AnalysisRepository()
