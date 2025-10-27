import psycopg2
import openai

# --- PostgreSQL connection setup ---
conn = psycopg2.connect(
    host="localhost",
    database="your_db",
    user="your_user",
    password="your_password"
)
cursor = conn.cursor()

# --- OpenAI API key ---
openai.api_key = "YOUR_OPENAI_API_KEY"

def search_annual_report(question, top_k=3):
    """
    Search report_data table for the most relevant pages based on the question.
    Returns concatenated text from top_k pages.
    """
    # Use PostgreSQL full-text search
    query = f"""
    SELECT page_text, ts_rank_cd(to_tsvector('english', page_text), plainto_tsquery('english', %s)) AS rank
    FROM report_data
    WHERE to_tsvector('english', page_text) @@ plainto_tsquery('english', %s)
    ORDER BY rank DESC
    LIMIT {top_k};
    """
    cursor.execute(query, (question, question))
    results = cursor.fetchall()
    
    # Combine top_k page texts
    combined_text = "\n\n".join([row[0] for row in results])
    return combined_text

def generate_answer(question, context_text):
    """
    Generate answer using OpenAI API, given context text.
    """
    prompt = f"""
You are a highly skilled financial analyst with 50+ years of experience. 
Use the provided annual report text to answer the question below as accurately and concisely as possible.
Always answer based on the text; do not hallucinate. Include numeric data where available.

Annual Report Text:
\"\"\"
{context_text}
\"\"\"

Question:
{question}

Answer:
"""
    response = openai.Completion.create(
        model="gpt-5-mini",
        prompt=prompt,
        max_tokens=400,
        temperature=0
    )
    return response.choices[0].text.strip()

# --- Example usage ---
question = "What was Tata Motors' consolidated revenue in FY25?"
context = search_annual_report(question)
answer = generate_answer(question, context)
print("Answer:", answer)

# --- Close connection ---
cursor.close()
conn.close()
