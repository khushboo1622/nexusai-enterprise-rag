"""
scripts/migrate_hr_to_mongo.py

Migrates HR employee data from CSV to MongoDB.
Run this ONCE before starting the server.

What it does:
  1. Reads your hr_data.csv from data/hr/
  2. Cleans and normalizes each row
  3. Inserts into MongoDB employees collection
  4. Creates indexes for fast querying

Usage:
    python -m scripts.migrate_hr_to_mongo

After running this:
  - Employee queries go through MCP tools -> MongoDB
  - HR policy questions still go through RAG -> Qdrant
  - You can re-run safely (upserts, not duplicates)
"""

import csv
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from backend.db.mongodb import get_db
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Fields to redact/hash before storing
SENSITIVE_FIELDS = ["salary", "wage", "compensation", "ssn", "tax_id"]

# Map common CSV column names to our standard field names
FIELD_MAPPING = {
    # Common variations -> our standard name
    "emp_id":           "employee_id",
    "empid":            "employee_id",
    "id":               "employee_id",
    "full_name":        "name",
    "employee_name":    "name",
    "dept":             "department",
    "job_title":        "role",
    "designation":      "role",
    "position":         "role",
    "joining_date":     "date_of_joining",
    "doj":              "date_of_joining",
    "dob":              "date_of_birth",
    "birth_date":       "date_of_birth",
    "manager":          "manager_id",
    "reporting_to":     "manager_id",
    "annual_leave":     "leave_balance",
    "leave_days":       "leave_balance",
    "sick_days":        "sick_leave_balance",
    "city":             "location",
    "office":           "location",
    "mobile":           "phone",
    "contact":          "phone",
    "active":           "status",
}


def normalize_row(row: dict) -> dict:
    """
    Normalize a CSV row into our standard employee schema.
    """
    normalized = {}

    for key, value in row.items():
        # Normalize key
        clean_key = key.lower().strip().replace(" ", "_").replace("-", "_")
        mapped_key = FIELD_MAPPING.get(clean_key, clean_key)

        # Normalize value
        clean_value = value.strip() if isinstance(value, str) else value

        # Redact sensitive fields
        if any(s in mapped_key for s in SENSITIVE_FIELDS):
            normalized[mapped_key] = "[REDACTED]"
        else:
            normalized[mapped_key] = clean_value

    # Ensure required fields exist
    if "employee_id" not in normalized:
        normalized["employee_id"] = f"EMP{str(uuid.uuid4())[:8].upper()}"

    if "status" not in normalized:
        normalized["status"] = "active"

    # Add metadata
    normalized["created_at"] = datetime.now(timezone.utc)
    normalized["source"] = "csv_migration"

    return normalized


def migrate():
    hr_dir = Path("data/hr")
    if not hr_dir.exists():
        logger.error("data/hr/ folder not found. Make sure you're running from project root.")
        return

    csv_files = list(hr_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No CSV files found in data/hr/")
        return

    db = get_db()
    col = db["employees"]

    # Create indexes
    col.create_index("employee_id", unique=True)
    col.create_index("name")
    col.create_index("department")
    col.create_index("status")
    logger.info("Indexes created")

    total_inserted = 0
    total_updated = 0
    total_failed = 0

    for csv_file in csv_files:
        logger.info(f"\nProcessing: {csv_file.name}")

        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                try:
                    doc = normalize_row(row)
                    emp_id = doc["employee_id"]

                    # Upsert — safe to re-run
                    result = col.update_one(
                        {"employee_id": emp_id},
                        {"$set": doc},
                        upsert=True,
                    )

                    if result.upserted_id:
                        total_inserted += 1
                    else:
                        total_updated += 1

                except Exception as e:
                    logger.warning(f"  Row {i+1} failed: {e}")
                    total_failed += 1

    total = total_inserted + total_updated
    logger.info(f"\nMigration complete!")
    logger.info(f"  Inserted : {total_inserted}")
    logger.info(f"  Updated  : {total_updated}")
    logger.info(f"  Failed   : {total_failed}")
    logger.info(f"  Total    : {total} employees in MongoDB")

    # Quick verification
    count = col.count_documents({})
    logger.info(f"\nVerification: {count} documents in employees collection")

    # Show sample
    sample = col.find_one({}, {"_id": 0, "employee_id": 1, "name": 1,
                                "department": 1, "role": 1, "status": 1})
    if sample:
        logger.info(f"Sample record: {sample}")


if __name__ == "__main__":
    migrate()