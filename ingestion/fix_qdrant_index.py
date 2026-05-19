"""
scripts/fix_qdrant_index.py

Creates missing payload indexes in Qdrant.
Run this ONCE to fix the doc_type filter error.

Usage:
    python -m scripts.fix_qdrant_index
"""

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from backend.config import get_settings

settings = get_settings()

client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
collection = settings.QDRANT_COLLECTION

print(f"Creating payload indexes on collection: {collection}")

indexes = [
    ("allowed_roles", PayloadSchemaType.KEYWORD),
    ("department",    PayloadSchemaType.KEYWORD),
    ("doc_type",      PayloadSchemaType.KEYWORD),
    ("content_type",  PayloadSchemaType.KEYWORD),
]

for field, schema in indexes:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
        print(f"  Created index: {field}")
    except Exception as e:
        print(f"  Skipped {field} (may already exist): {e}")

print("Done!")