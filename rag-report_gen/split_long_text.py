import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection setup
DB_NAME = "jkinda_stocks"
DB_USER = "ashwani_ocr"
DB_PASSWORD = "ashwani"
DB_HOST = "localhost"
DB_PORT = "5432"

# Table and column names
TABLE_NAME = "financial_excerpts"
TEXT_COLUMN = "excerpt"
FILENAME_COLUMN = "filename"
MAX_WORDS = 400


def split_text_into_chunks(text, max_words=MAX_WORDS):
    """Split text into smaller chunks with at most `max_words` words."""
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def main():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Fetch all excerpts and filenames
    cur.execute(f"SELECT id, {TEXT_COLUMN}, {FILENAME_COLUMN} FROM {TABLE_NAME};")
    rows = cur.fetchall()

    for row in rows:
        excerpt_id = row["id"]
        excerpt = row[TEXT_COLUMN]
        filename = row.get(FILENAME_COLUMN)

        if not excerpt or not isinstance(excerpt, str):
            continue

        words = excerpt.split()
        word_count = len(words)

        if word_count <= MAX_WORDS:
            continue  # Already short enough

        # Split into smaller chunks
        chunks = split_text_into_chunks(excerpt, MAX_WORDS)
        print(f"Splitting ID {excerpt_id} ({filename}) into {len(chunks)} chunks ({word_count} words total).")

        # Update the first record with the first chunk
        cur.execute(
            f"UPDATE {TABLE_NAME} SET {TEXT_COLUMN} = %s WHERE id = %s;",
            (chunks[0], excerpt_id)
        )

        # Insert remaining chunks as new records, preserving filename
        for chunk in chunks[1:]:
            cur.execute(
                f"INSERT INTO {TABLE_NAME} ({TEXT_COLUMN}, {FILENAME_COLUMN}) VALUES (%s, %s);",
                (chunk, filename)
            )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Long excerpts split and updated successfully (filenames preserved).")


if __name__ == "__main__":
    main()
