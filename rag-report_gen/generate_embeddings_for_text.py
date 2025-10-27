import psycopg2
from sentence_transformers import SentenceTransformer

# ====== CONFIGURATION ======
DB_CONFIG = {
    "dbname": "jkinda_stocks",
    "user": "ashwani_ocr",
    "password": "ashwani",
    "host": "localhost",
    "port": "5432"
}

TABLE_NAME = "financial_excerpts"

# ====== LOAD MODEL ======
# You can switch to another if desired (MiniLM is faster, E5 gives higher quality)
model = SentenceTransformer("intfloat/e5-large-v2")

# ====== CONNECT TO DATABASE ======
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# ====== FETCH EXCERPTS WITHOUT EMBEDDINGS ======
cur.execute(f"SELECT id, excerpt FROM {TABLE_NAME} WHERE embedding IS NULL;")
rows = cur.fetchall()
print(f"📄 Found {len(rows)} excerpts without embeddings")

# ====== GENERATE AND STORE EMBEDDINGS ======
i = 0
for record_id, excerpt in rows:
    # For E5 models, prepend "passage:" to text for better results
    text = f"passage: {excerpt}"
    embedding = model.encode(text).tolist()
    
    cur.execute(f"UPDATE {TABLE_NAME} SET embedding = %s WHERE id = %s;", (embedding, record_id))
    i += 1
    if i % 100 == 0:
        print(f"  ✅ Processed {i} excerpts")
        break


conn.commit()
cur.close()
conn.close()

print("✅ Embeddings generated and stored successfully!")
