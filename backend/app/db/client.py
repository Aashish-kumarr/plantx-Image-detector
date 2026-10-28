"""MongoDB client singleton for PlantDx backend."""

from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database

from app.core.logging import get_logger

logger = get_logger("db.client")

_client: Optional[MongoClient] = None
_database: Optional[Database] = None


def get_client() -> MongoClient:
    """Return the active MongoClient instance.

    Raises RuntimeError if the client has not been initialized via connect().
    """
    if _client is None:
        raise RuntimeError(
            "MongoDB client is not initialized. "
            "Ensure connect() is called during application startup."
        )
    return _client


def get_database() -> Database:
    """Return the active plantdx Database instance."""
    if _database is None:
        raise RuntimeError(
            "MongoDB database is not initialized. "
            "Ensure connect() is called during application startup."
        )
    return _database


def connect(uri: str, db_name: str) -> None:
    """Create a MongoClient, verify connectivity, and store singletons.

    Args:
        uri: MongoDB connection URI.
        db_name: Name of the database to select.
    """
    global _client, _database

    logger.info("Connecting to MongoDB (database=%s)...", db_name)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # Verify the connection is reachable before accepting traffic.
    client.admin.command("ping")

    _client = client
    _database = client[db_name]
    logger.info("MongoDB connection established.")


def disconnect() -> None:
    """Close the MongoClient connection gracefully."""
    global _client, _database

    if _client is not None:
        logger.info("Closing MongoDB connection...")
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed.")


def is_connected() -> bool:
    """Return True if the client singleton is initialized."""
    return _client is not None
