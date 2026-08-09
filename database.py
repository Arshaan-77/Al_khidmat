import psycopg2
import numpy as np

import os
from psycopg2.extras import Json
from dotenv import load_dotenv

from face_embeddings import (
    generate_embedding,
    check_duplicate_face
)

load_dotenv()

# ==========================
# PostgreSQL Configuration
# ==========================

hostname = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
username = os.getenv("DB_USER")
pwd = os.getenv("DB_PASSWORD")
port_id = os.getenv("DB_PORT")


# ==========================
# Database Connection
# ==========================

def get_connection():

    return psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=pwd,
        port=port_id
    )


# ==========================
# Create Tables
# ==========================

def create_tables():

    conn = None
    cur = None

    try:

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS beneficiary (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            phone VARCHAR(20) NOT NULL,
            location VARCHAR(255),
            family_members INT,
            need VARCHAR(100),
            need_cost INT,
            need_details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS beneficiary_face (
            id SERIAL PRIMARY KEY,
            beneficiary_id INT REFERENCES beneficiary(id)
            ON DELETE CASCADE,
            image_path TEXT NOT NULL,
            embedding JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()

    except Exception as e:

        print("Table Creation Error:", e)

    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# ==========================
# Save Beneficiary
# ==========================

def save_beneficiaries2(data):

    conn = None
    cur = None

    try:

        create_tables()

        conn = get_connection()
        cur = conn.cursor()

        beneficiary = data.get(
            "beneficiary",
            {}
        )

        need_type = str(
            beneficiary.get(
                "need",
                ""
            )
        ).lower()

        if "medical" in need_type:

            need_details = data.get(
                "medical",
                {}
            )

        elif "education" in need_type:

            need_details = data.get(
                "education",
                {}
            )

        elif "financial" in need_type:

            need_details = data.get(
                "financial",
                {}
            )

        else:

            need_details = {}

        cur.execute(
            """
            INSERT INTO beneficiary(
                name,
                phone,
                location,
                family_members,
                need,
                need_cost,
                need_details
            )
            VALUES(
                %s,%s,%s,%s,%s,%s,%s
            )
            RETURNING id
            """,
            (
                beneficiary.get("name"),
                beneficiary.get("phone"),
                beneficiary.get("location"),
                beneficiary.get("family_members"),
                beneficiary.get("need"),
                beneficiary.get("need_cost"),
                Json(need_details)
            )
        )

        beneficiary_id = cur.fetchone()[0]

        conn.commit()

        print(
            f"Beneficiary Saved: {beneficiary_id}"
        )

        return beneficiary_id

    except Exception as e:

        print(
            "Beneficiary Save Error:",
            e
        )

        if conn:
            conn.rollback()  

        return None

    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# ==========================
# Load Existing Embeddings
# ==========================

def get_all_embeddings():

    conn = None
    cur = None

    try:

        create_tables()

        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        SELECT
            beneficiary_id,
            embedding
        FROM beneficiary_face
        """)

        rows = cur.fetchall()

        embeddings = []

        for row in rows:

            embeddings.append(
                {
                    "beneficiary_id": row[0],
                    "embedding": row[1]
                }
            )

        return embeddings

    except Exception as e:

        print(
            "Embedding Fetch Error:",
            e
        )

        return []

    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# ==========================
# Check Duplicate Face
# ==========================

def check_duplicate_in_database(
    image_path,
    threshold=0.80
):

    try:

        new_embedding = generate_embedding(
            image_path
        )

        existing_embeddings = (
            get_all_embeddings()
        )

        result = check_duplicate_face(
            new_embedding,
            existing_embeddings,
            threshold
        )

        return result

    except Exception as e:

        return {
            "error": str(e)
        }


# ==========================
# Save Face Embedding
# ==========================

def save_face_embedding(
    beneficiary_id,
    image_path
):

    conn = None
    cur = None

    try:

        create_tables()

        embedding = generate_embedding(
            image_path
        )

        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO beneficiary_face(
                beneficiary_id,
                image_path,
                embedding
            )
            VALUES(
                %s,%s,%s
            )
            """,
            (
                beneficiary_id,
                image_path,
                Json(embedding)
            )
        )

        conn.commit()

        return {
            "success": True
        }

    except Exception as e:

        print(
            "Embedding Save Error:",
            e
        )

        if conn:
            conn.rollback()

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# ==========================
# Get Beneficiary By ID
# ==========================

def get_beneficiary(
    beneficiary_id
):

    conn = None
    cur = None

    try:

        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM beneficiary
            WHERE id=%s
            """,
            (
                beneficiary_id,
            )
        )

        row = cur.fetchone()

        return row

    except Exception as e:

        print(
            "Fetch Error:",
            e
        )

        return None

    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()