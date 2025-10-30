import psycopg2
from sentence_transformers import SentenceTransformer

# This will only run once to generate embeddings for questions

# Load a free open-source embedding model (you can swap with others below)
model = SentenceTransformer("intfloat/e5-large-v2")
# Other good free models:
# "all-MiniLM-L6-v2"
# "nomic-ai/nomic-embed-text-v1.5"
# "thenlper/gte-large"

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=jkinda_stocks user=ashwani_ocr password=ashwani host=localhost port=5432")
cur = conn.cursor()

# Fetch records without embeddings
cur.execute("SELECT id, question FROM rag_questions WHERE embedding IS NULL;")
rows = cur.fetchall()

for id, question in rows:
    # For E5 models, prepend "query:" for better embedding performance
    text = f"query: {question}"
    embedding = model.encode(text).tolist()
    
    # Update the embedding column (VECTOR type expects list of floats)
    cur.execute("UPDATE rag_questions SET embedding = %s WHERE id = %s;", (embedding, id))

# Commit changes
conn.commit()
cur.close()
conn.close()

print("✅ Embeddings generated and stored successfully!")
