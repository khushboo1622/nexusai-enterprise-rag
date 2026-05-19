"""
backend/db/mongodb.py

Central MongoDB connection module.
All other modules import `db` from here — one connection, reused everywhere.

Collections:
  - users         : stores user accounts (email, hashed password, role)
  - chat_logs     : silently stores all queries for Ragas eval (Phase 2)
"""

import logging
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Create one MongoClient for the entire app lifetime ─────────────────────
# MongoClient is thread-safe and manages a connection pool internally
# Never create a new MongoClient per request — expensive and slow
_client: MongoClient = None
_db: Database = None


def get_db() -> Database:
    """
    Returns the MongoDB database instance.
    Creates connection on first call, reuses on subsequent calls.
    """
    global _client, _db

    if _db is None:
        try:
            _client = MongoClient(settings.MONGO_URI)
            _db = _client[settings.MONGO_DB_NAME]
            # Ping to verify connection is actually working
            _client.admin.command("ping")
            logger.info(f"MongoDB connected → database: '{settings.MONGO_DB_NAME}' ✓")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    return _db


def get_users_collection() -> Collection:
    """Returns the users collection."""
    return get_db()["users"]


def get_chat_logs_collection() -> Collection:
    """
    Returns the chat_logs collection.
    Used in Phase 2 to silently log all queries for Ragas evaluation.
    """
    return get_db()["chat_logs"]


def close_db():
    """Close MongoDB connection. Called on app shutdown."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed.")