"""
NGO Registration Assistant — Database Module
=============================================
PostgreSQL schema with 4 tables:
  - beneficiaries        (common registration fields)
  - medical_needs        (medical-specific fields)
  - education_needs      (education-specific fields)
  - financial_needs      (financial-specific fields)

Each need table has a foreign key → beneficiaries.id
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONNECTION CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "dbname":   os.getenv("DB_NAME",     "langgraph_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "@rshaan786"),
    "port":     int(os.getenv("DB_PORT", 5432)),
}


def _get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
# TABLE CREATION
# ──────────────────────────────────────────────────────────────────────────────

_CREATE_BENEFICIARIES = """
CREATE TABLE IF NOT EXISTS beneficiaries (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100)    NOT NULL,
    phone           VARCHAR(20)     NOT NULL,
    location        VARCHAR(100),
    family_members  INT,
    need_category   VARCHAR(20)     NOT NULL,   -- 'medical' | 'education' | 'financial'
    registered_at   TIMESTAMP       DEFAULT NOW()
);
"""

_CREATE_MEDICAL_NEEDS = """
CREATE TABLE IF NOT EXISTS medical_needs (
    id                  SERIAL PRIMARY KEY,
    beneficiary_id      INT NOT NULL REFERENCES beneficiaries(id) ON DELETE CASCADE,
    disease             VARCHAR(100),
    hospital            VARCHAR(150),
    urgency             VARCHAR(10),            -- 'high' | 'medium' | 'low'
    need_cost           INT
);
"""

_CREATE_EDUCATION_NEEDS = """
CREATE TABLE IF NOT EXISTS education_needs (
    id                  SERIAL PRIMARY KEY,
    beneficiary_id      INT NOT NULL REFERENCES beneficiaries(id) ON DELETE CASCADE,
    student_class       VARCHAR(50),
    institute           VARCHAR(150),
    academic_status     VARCHAR(50)             -- 'passed' | 'failed' | 'appearing'
);
"""

_CREATE_FINANCIAL_NEEDS = """
CREATE TABLE IF NOT EXISTS financial_needs (
    id                  SERIAL PRIMARY KEY,
    beneficiary_id      INT NOT NULL REFERENCES beneficiaries(id) ON DELETE CASCADE,
    monthly_income      INT,
    employment_status   VARCHAR(50),
    earning_members     INT
);
"""




def create_tables():
    """
    Create all tables, handling migration from the old single-table schema.
    The old schema had a beneficiaries table with no id column.
    If detected, drops all old tables and recreates with correct schema.
    """
    conn = _get_connection()
    cur  = conn.cursor()
    try:
        # Detect old schema: beneficiaries exists but has no id column
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'beneficiaries'
              AND column_name = 'id'
        """)
        has_id_column = cur.fetchone() is not None

        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'beneficiaries'
            )
        """)
        table_exists = cur.fetchone()[0]

        if table_exists and not has_id_column:
            # Old schema — drop everything and start fresh
            print("[DB] Old schema detected (no id column). Migrating...")
            cur.execute("DROP TABLE IF EXISTS financial_needs  CASCADE")
            cur.execute("DROP TABLE IF EXISTS education_needs  CASCADE")
            cur.execute("DROP TABLE IF EXISTS medical_needs    CASCADE")
            cur.execute("DROP TABLE IF EXISTS beneficiaries    CASCADE")
            print("[DB] Old tables dropped.")

        # Create all tables with correct schema
        cur.execute(_CREATE_BENEFICIARIES)
        cur.execute(_CREATE_MEDICAL_NEEDS)
        cur.execute(_CREATE_EDUCATION_NEEDS)
        cur.execute(_CREATE_FINANCIAL_NEEDS)
        conn.commit()
        print("[DB] Tables ready.")
    except Exception as e:
        conn.rollback()
        print(f"[DB] Error creating tables: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# SAVE REGISTRATION
# ──────────────────────────────────────────────────────────────────────────────

def save_registration(data: dict) -> int:
    """
    Save a completed registration to the database.

    Expects `data` in the same format as `extracted_dict` from BotState:
    {
        "name": "Arshaan",
        "phone": "9876543210",
        "location": "Bhopal",
        "family_members": 5,
        "medical": {
            "disease": "cancer",
            "hospital": "Hamidia Hospital",
            "urgency": "high",
            "need_cost": 250000
        }
        # OR "education": {...}
        # OR "financial": {...}
    }

    Returns the new beneficiary ID.
    """

    # ── Determine active category ──────────────────────────────────────────
    active_category = None
    for cat in ("medical", "education", "financial"):
        if data.get(cat):
            active_category = cat
            break

    if not active_category:
        raise ValueError("No need category found in data. Cannot save registration.")

    conn = _get_connection()
    cur  = conn.cursor()

    try:
        # ── 1. Insert into beneficiaries ───────────────────────────────────
        cur.execute(
            """
            INSERT INTO beneficiaries (name, phone, location, family_members, need_category)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                data.get("name"),
                data.get("phone"),
                data.get("location"),
                data.get("family_members"),
                active_category,
            ),
        )
        beneficiary_id = cur.fetchone()[0]

        # ── 2. Insert into the matching need table ─────────────────────────
        need_data = data[active_category]

        if active_category == "medical":
            cur.execute(
                """
                INSERT INTO medical_needs (beneficiary_id, disease, hospital, urgency, need_cost)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    beneficiary_id,
                    need_data.get("disease"),
                    need_data.get("hospital"),
                    need_data.get("urgency"),
                    need_data.get("need_cost"),
                ),
            )

        elif active_category == "education":
            cur.execute(
                """
                INSERT INTO education_needs (beneficiary_id, student_class, institute, academic_status)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    beneficiary_id,
                    need_data.get("student_class"),
                    need_data.get("institute"),
                    need_data.get("academic_status"),
                ),
            )

        elif active_category == "financial":
            cur.execute(
                """
                INSERT INTO financial_needs (beneficiary_id, monthly_income, employment_status, earning_members)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    beneficiary_id,
                    need_data.get("monthly_income"),
                    need_data.get("employment_status"),
                    need_data.get("earning_members"),
                ),
            )

        conn.commit()
        print(f"[DB] Registration saved. Beneficiary ID: {beneficiary_id}")
        return beneficiary_id

    except Exception as e:
        conn.rollback()
        print(f"[DB] Error saving registration: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# READ HELPERS  (optional — useful for admin/review)
# ──────────────────────────────────────────────────────────────────────────────

def get_all_beneficiaries() -> list[dict]:
    """Return all beneficiaries with their need details as a flat joined record."""
    conn = _get_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            SELECT
                b.id,
                b.name,
                b.phone,
                b.location,
                b.family_members,
                b.need_category,
                b.registered_at,

                -- medical
                m.disease,
                m.hospital,
                m.urgency,
                m.need_cost,

                -- education
                e.student_class,
                e.institute,
                e.academic_status,

                -- financial
                f.monthly_income,
                f.employment_status,
                f.earning_members

            FROM beneficiaries b
            LEFT JOIN medical_needs    m ON m.beneficiary_id = b.id
            LEFT JOIN education_needs  e ON e.beneficiary_id = b.id
            LEFT JOIN financial_needs  f ON f.beneficiary_id = b.id
            ORDER BY b.registered_at DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def get_beneficiary_by_id(beneficiary_id: int) -> dict | None:
    """Return a single beneficiary record with need details."""
    conn = _get_connection()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            SELECT
                b.id, b.name, b.phone, b.location, b.family_members,
                b.need_category, b.registered_at,
                m.disease, m.hospital, m.urgency, m.need_cost,
                e.student_class, e.institute, e.academic_status,
                f.monthly_income, f.employment_status, f.earning_members
            FROM beneficiaries b
            LEFT JOIN medical_needs    m ON m.beneficiary_id = b.id
            LEFT JOIN education_needs  e ON e.beneficiary_id = b.id
            LEFT JOIN financial_needs  f ON f.beneficiary_id = b.id
            WHERE b.id = %s
            """,
            (beneficiary_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()
        conn.close()