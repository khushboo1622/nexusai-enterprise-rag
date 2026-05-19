"""
ingestion/ingest.py

Run this ONCE before starting the backend server.
This script:
  1. Reads all files from data/ folder (organized by department)
  2. Chunks them using LlamaIndex SentenceSplitter
  3. Creates embeddings using BGE-small (runs locally, no API key needed)
  4. Pushes everything to Qdrant with rich metadata

Usage:
    python -m ingestion.ingest

Folder structure expected:
    data/
    ├── engineering/   → allowed_roles: [ENGINEERING, C_LEVEL]
    ├── marketing/     → allowed_roles: [MARKETING, C_LEVEL]
    ├── finance/       → allowed_roles: [FINANCE, C_LEVEL]
    ├── hr/            → allowed_roles: [HR, C_LEVEL]
    └── general/       → allowed_roles: [HR, FINANCE, ENGINEERING, MARKETING, C_LEVEL]
"""

import os
import sys
import uuid
import csv
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── LlamaIndex ─────────────────────────────────────────────────────────────
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Qdrant ─────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
# BGE-small produces 384-dimensional vectors
VECTOR_SIZE = 384

# Which roles can access which department's documents
# general is accessible to everyone
DEPARTMENT_ROLES: dict[str, list[str]] = {
    "engineering": ["ENGINEERING", "C_LEVEL"],
    "marketing":   ["MARKETING",   "C_LEVEL"],
    "finance":     ["FINANCE",     "C_LEVEL"],
    "hr":          ["HR",          "C_LEVEL"],
    # General docs accessible to ALL roles including GENERAL
    # GENERAL = operations, compliance, risk, QA etc.
    "general":     ["HR", "FINANCE", "ENGINEERING", "MARKETING", "C_LEVEL", "GENERAL"],
}

# doc_type mapping by department
DEPARTMENT_DOC_TYPE: dict[str, str] = {
    "engineering": "policy",
    "marketing":   "strategy",
    "finance":     "report",
    "hr":          "employee_record",
    "general":     "info",
}


# ── Helper: load and chunk a markdown file ─────────────────────────────────
def load_markdown_file(filepath: Path, department: str) -> list[Document]:
    """
    Read a .md file and return a list of LlamaIndex Documents.
    We create ONE Document per file — LlamaIndex splitter will chunk it.
    """
    text = filepath.read_text(encoding="utf-8")

    # LlamaIndex Document wraps raw text + metadata
    # metadata will be attached to every chunk produced from this document
    doc = Document(
        text=text,
        metadata={
            "source_file": str(filepath.relative_to(Path("data"))),
            "department": department.upper(),
            "allowed_roles": DEPARTMENT_ROLES[department],
            "doc_type": DEPARTMENT_DOC_TYPE[department],
            "content_type": "markdown",
            "file_name": filepath.name,
        }
    )
    return [doc]


# ── Helper: load and chunk a CSV file ─────────────────────────────────────
def load_csv_file(filepath: Path, department: str) -> list[Document]:
    """
    Read a .csv file and convert each row into a Document.
    Each row = one employee record = one searchable chunk.
    We convert row to natural language text for better embedding quality.
    Salary fields are REDACTED for security.
    """
    documents = []

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Redact sensitive fields before ingestion
            # Even if RBAC prevents access, we add defense-in-depth
            redacted_row = {}
            for key, value in row.items():
                key_lower = key.lower()
                if any(s in key_lower for s in ["salary", "wage", "pay", "compensation", "ssn", "tax"]):
                    redacted_row[key] = "[REDACTED]"
                else:
                    redacted_row[key] = value

            # Convert row dict to natural language string
            # "Name: John Doe | Role: HR Manager | Department: HR | ..."
            text = " | ".join(
                f"{k}: {v}" for k, v in redacted_row.items() if v.strip()
            )

            doc = Document(
                text=text,
                metadata={
                    "source_file": str(filepath.relative_to(Path("data"))),
                    "department": department.upper(),
                    "allowed_roles": DEPARTMENT_ROLES[department],
                    "doc_type": DEPARTMENT_DOC_TYPE[department],
                    "content_type": "csv",
                    "file_name": filepath.name,
                    "row_index": idx,
                }
            )
            documents.append(doc)

    logger.info(f"  Loaded {len(documents)} rows from {filepath.name}")
    return documents


# ── Main ingestion function ────────────────────────────────────────────────
def run_ingestion():
    # ── Step 1: Load settings from .env ───────────────────────────────────
    qdrant_url    = os.getenv("QDRANT_URL")
    qdrant_key    = os.getenv("QDRANT_API_KEY")
    collection    = os.getenv("QDRANT_COLLECTION", "nexusai_docs")
    embed_model   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    data_dir      = Path("data")

    if not qdrant_url or not qdrant_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    if not data_dir.exists():
        logger.error(f"data/ folder not found. Create it and add your files.")
        sys.exit(1)

    # ── Step 2: Initialize embedding model ────────────────────────────────
    # HuggingFaceEmbedding downloads BGE-small on first run (~30MB)
    # Subsequent runs use cached model — fast
    logger.info(f"Loading embedding model: {embed_model}")
    embed = HuggingFaceEmbedding(
        model_name=embed_model,
        # normalize embeddings = better cosine similarity scores
        embed_batch_size=32,
    )
    logger.info("Embedding model loaded ✓")

    # ── Step 3: Connect to Qdrant ──────────────────────────────────────────
    logger.info(f"Connecting to Qdrant at {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Create collection if it doesn't exist
    # distance=COSINE because BGE-small is trained with cosine similarity
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        logger.info(f"Creating collection '{collection}'")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,      # 384 for BGE-small
                distance=Distance.COSINE,
            ),
        )
        # Create payload index on 'allowed_roles' for fast RBAC filtering
        # Without this index, every query scans all points — slow at scale
        client.create_payload_index(
            collection_name=collection,
            field_name="allowed_roles",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        # Index department too for potential future filtering
        client.create_payload_index(
            collection_name=collection,
            field_name="department",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Collection '{collection}' created with indexes ✓")
    else:
        logger.info(f"Collection '{collection}' already exists ✓")

    # ── Step 4: Load all documents ─────────────────────────────────────────
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    all_points: list[PointStruct] = []

    for department in DEPARTMENT_ROLES.keys():
        dept_dir = data_dir / department
        if not dept_dir.exists():
            logger.warning(f"  Folder data/{department}/ not found — skipping")
            continue

        files = list(dept_dir.iterdir())
        if not files:
            logger.warning(f"  data/{department}/ is empty — skipping")
            continue

        logger.info(f"\nProcessing department: {department.upper()}")

        for filepath in files:
            if filepath.suffix == ".md":
                logger.info(f"  Reading {filepath.name}")
                docs = load_markdown_file(filepath, department)

            elif filepath.suffix == ".csv":
                logger.info(f"  Reading {filepath.name}")
                docs = load_csv_file(filepath, department)

            else:
                logger.warning(f"  Skipping unsupported file: {filepath.name}")
                continue

            # ── Step 5: Chunk documents ────────────────────────────────────
            # SentenceSplitter splits on sentence boundaries — cleaner chunks
            # than arbitrary character splitting
            # For CSV docs (already one sentence per row), this is a no-op
            nodes = splitter.get_nodes_from_documents(docs)
            logger.info(f"  Chunked into {len(nodes)} nodes")

            # ── Step 6: Create embeddings + build Qdrant points ────────────
            for idx, node in enumerate(nodes):
                # Get the text content of this chunk
                text = node.get_content()
                if not text.strip():
                    continue

                # Create embedding vector using BGE-small
                vector = embed.get_text_embedding(text)

                # Build the Qdrant point
                # payload = everything you want to store + filter on later
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        # Core content
                        "text": text,
                        # RBAC — used to filter at query time
                        "allowed_roles": node.metadata.get("allowed_roles", []),
                        # Metadata for citations and debugging
                        "department":    node.metadata.get("department", ""),
                        "source_file":   node.metadata.get("source_file", ""),
                        "file_name":     node.metadata.get("file_name", ""),
                        "doc_type":      node.metadata.get("doc_type", ""),
                        "content_type":  node.metadata.get("content_type", ""),
                        "chunk_index":   idx,
                        # row_index only present for CSV rows
                        "row_index":     node.metadata.get("row_index", None),
                    }
                )
                all_points.append(point)

    if not all_points:
        logger.error("No points created. Check your data/ folder.")
        sys.exit(1)

    # ── Step 7: Upload to Qdrant in batches ───────────────────────────────
    # Batch upload is faster than one-by-one
    # 100 points per batch is safe for free tier
    BATCH_SIZE = 100
    total = len(all_points)
    logger.info(f"\nUploading {total} points to Qdrant in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = all_points[i:i + BATCH_SIZE]
        client.upsert(
            collection_name=collection,
            points=batch,
        )
        logger.info(f"  Uploaded {min(i + BATCH_SIZE, total)}/{total}")

    logger.info(f"\n✅ Ingestion complete!")
    logger.info(f"   Collection : {collection}")
    logger.info(f"   Total points uploaded: {total}")
    logger.info(f"\nBreakdown by department:")

    # Print summary
    dept_counts: dict[str, int] = {}
    for p in all_points:
        dept = p.payload.get("department", "UNKNOWN")
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    for dept, count in sorted(dept_counts.items()):
        logger.info(f"   {dept:<15} : {count} chunks")


if __name__ == "__main__":
    run_ingestion()