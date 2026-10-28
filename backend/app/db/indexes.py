"""MongoDB index initialization for PlantDx collections."""

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from app.core.logging import get_logger

logger = get_logger("db.indexes")


def create_indexes(db: Database) -> None:
    """Create required indexes on all PlantDx collections.

    Safe to call on every startup — pymongo skips creation if the index
    already exists with the same key pattern and options.
    """
    analyses = db["analyses"]

    # Unique index on the application-level UUID identifier.
    analyses.create_index([("id", ASCENDING)], unique=True, name="analyses_id_unique")

    # Index supporting newest-first history ordering.
    analyses.create_index(
        [("created_at", DESCENDING)], name="analyses_created_at_desc"
    )

    # Index supporting per-user history queries.
    analyses.create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)], name="analyses_user_id_created_at"
    )


    logger.info("MongoDB indexes verified.")
