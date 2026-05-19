"""
scripts/setup_employee_auth.py

Sets up authentication for all employees in MongoDB.
Run this ONCE after migrate_hr_to_mongo.py.

What it does:
  1. Reads all employees from MongoDB
  2. Generates password: EMP + last4 of employee_id + @Nexus
  3. Hashes it with bcrypt
  4. Stores hashed_password back into employee record

Usage:
    python -m scripts.setup_employee_auth

After running:
  Employees can login with:
    employee_id: FINEMP1012
    password:    EMP1012@Nexus
"""

import logging
from dotenv import load_dotenv
load_dotenv()

from passlib.context import CryptContext
from backend.db.mongodb import get_db
from backend.auth.utils import generate_employee_password

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def setup():
    db = get_db()
    col = db["employees"]

    total = col.count_documents({})
    logger.info(f"Found {total} employees in MongoDB")

    updated = 0
    skipped = 0

    for emp in col.find({}, {"employee_id": 1, "name": 1, "hashed_password": 1}):
        emp_id = emp.get("employee_id", "")
        if not emp_id:
            skipped += 1
            continue

        # Skip if password already set
        if emp.get("hashed_password"):
            skipped += 1
            continue

        plain_password = generate_employee_password(emp_id)
        hashed = pwd_context.hash(plain_password)

        col.update_one(
            {"employee_id": emp_id},
            {"$set": {"hashed_password": hashed}}
        )
        updated += 1

    logger.info(f"Done! Updated: {updated} | Skipped (already set): {skipped}")

    # Show sample credentials
    logger.info("\nSample credentials:")
    for emp in col.find({}, {"employee_id": 1, "name": 1, "department": 1}).limit(5):
        emp_id = emp.get("employee_id", "")
        pwd = generate_employee_password(emp_id)
        logger.info(f"  {emp_id} ({emp.get('name')}, {emp.get('department')}) -> {pwd}")


if __name__ == "__main__":
    setup()