import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()



DB_URI = "postgresql://postgres:%40rshaan786@localhost:5432/langgraph_db"

hostname = "localhost"
database = "langgraph_db"
username= "postgres"
pwd = "@rshaan786"
port_id = 5432
conn = None
cur = None

def save_beneficiaries(data):
    conn = psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=pwd,
        port=port_id)
    
    cur = conn.cursor()

    create_script = """
        CREATE TABLE IF NOT EXISTS beneficiaries (
            name varchar(30) NOT NULL,
            phone varchar(14) NOT NULL,
            location varchar(100),
            family_members int,
            need varchar(200) NOT NULL
        )
"""

    cur.execute(create_script)
    
    insert_script = '''INSERT INTO beneficiaries (
        name,
        phone, 
        location, 
        family_members, 
        need)
        VALUES (%s, %s, %s, %s, %s)
    '''
    
    insert_values = data.get("name"), data.get("phone"), data.get("location"), data.get("family_members"), data.get("need")

    cur.execute(insert_script, insert_values)

    conn.commit()

    cur.close()
    conn.close()

