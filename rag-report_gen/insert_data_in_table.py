import os
import psycopg2

# ====== CONFIGURATION ======
DB_CONFIG = {
    "dbname": "jkinda_stocks",
    "user": "ashwani_ocr",
    "password": "ashwani",
    "host": "localhost",
    "port": "5432"
}

# make this dynamic as per your needs
INPUT_DIR = "output/content/Lg-el"
DELIMITER = "---||---"
TABLE_NAME = "financial_excerpts"

# ====== CONNECT TO DATABASE ======
def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to PostgreSQL")
        return conn
    except Exception as e:
        print("❌ Database connection failed:", e)
        exit(1)

# ====== INSERT FUNCTION ======
def insert_excerpt(cursor, filename, excerpt_text):
    query = f"""
        INSERT INTO {TABLE_NAME} (filename, excerpt)
        VALUES (%s, %s);
    """
    cursor.execute(query, (filename, excerpt_text))

# ====== MAIN SCRIPT ======
def main():
    conn = get_connection()
    cursor = conn.cursor()

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(INPUT_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        excerpts = [chunk.strip() for chunk in content.split(DELIMITER) if chunk.strip()]
        print(f"📄 Processing {filename}: found {len(excerpts)} excerpts")

        for excerpt in excerpts:
            insert_excerpt(cursor, filename, excerpt)

        conn.commit()
        print(f"✅ Inserted {len(excerpts)} excerpts from {filename}")

    cursor.close()
    conn.close()
    print("🎯 All files processed successfully!")

if __name__ == "__main__":
    main()
