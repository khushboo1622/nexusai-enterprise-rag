"""
scripts/seed_users.py

Creates test users in MongoDB with properly hashed passwords.
Run this ONCE after setting up MongoDB.

Usage:
    python -m scripts.seed_users
"""

import sys
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from pymongo import MongoClient
from passlib.context import CryptContext
from backend.config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TEST_USERS = [
    # HR users
    {
        "name":     "Priya Sharma",
        "email":    "priya.sharma@nexusai.com",
        "password": "hr_priya_123",
        "role":     "HR",
    },
    {
        "name":     "Rahul Mehta",
        "email":    "rahul.mehta@nexusai.com",
        "password": "hr_rahul_123",
        "role":     "HR",
    },
    # Finance users
    {
        "name":     "Anita Patel",
        "email":    "anita.patel@nexusai.com",
        "password": "fin_anita_123",
        "role":     "FINANCE",
    },
    {
        "name":     "Vikram Joshi",
        "email":    "vikram.joshi@nexusai.com",
        "password": "fin_vikram_123",
        "role":     "FINANCE",
    },
    # Engineering users
    {
        "name":     "Sneha Reddy",
        "email":    "sneha.reddy@nexusai.com",
        "password": "eng_sneha_123",
        "role":     "ENGINEERING",
    },
    {
        "name":     "Arjun Kumar",
        "email":    "arjun.kumar@nexusai.com",
        "password": "eng_arjun_123",
        "role":     "ENGINEERING",
    },
    # Marketing users
    {
        "name":     "Meera Iyer",
        "email":    "meera.iyer@nexusai.com",
        "password": "mkt_meera_123",
        "role":     "MARKETING",
    },
    {
        "name":     "Rohan Gupta",
        "email":    "rohan.gupta@nexusai.com",
        "password": "mkt_rohan_123",
        "role":     "MARKETING",
    },
    # C-Level users
    {
        "name":     "Kavya Nair",
        "email":    "kavya.nair@nexusai.com",
        "password": "clevel_kavya_123",
        "role":     "C_LEVEL",
    },
    {
        "name":     "Amit Shah",
        "email":    "amit.shah@nexusai.com",
        "password": "clevel_amit_123",
        "role":     "C_LEVEL",
    },
]


def seed():
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    users = db["users"]

    # Create unique index on email — prevents duplicates
    users.create_index("email", unique=True)

    created = 0
    skipped = 0

    print("\n🌱 Seeding test users...\n")

    for user_data in TEST_USERS:
        existing = users.find_one({"email": user_data["email"]})
        if existing:
            print(f"  ⚠️  Skipping (already exists): {user_data['email']}")
            skipped += 1
            continue

        user_doc = {
            "user_id":         str(uuid.uuid4()),
            "name":            user_data["name"],
            "email":           user_data["email"],
            "hashed_password": pwd_context.hash(user_data["password"]),
            "role":            user_data["role"],
            "created_at":      datetime.now(timezone.utc),
        }
        users.insert_one(user_doc)
        print(f"  ✅ Created: {user_data['email']} | role: {user_data['role']}")
        created += 1

    print(f"\n✅ Done! Created: {created} | Skipped: {skipped}")
    client.close()


if __name__ == "__main__":
    seed()