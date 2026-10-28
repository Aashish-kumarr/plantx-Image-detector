"""app/db package — MongoDB access layer."""

from app.db.client import connect, disconnect, get_client, get_database, is_connected

__all__ = ["connect", "disconnect", "get_client", "get_database", "is_connected"]
