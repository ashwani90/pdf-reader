Got it! You want a Python workflow that:

1. Takes a **question**.
2. Searches your `report_data` PostgreSQL table for relevant pages (full-text search).
3. Feeds the relevant text and the question to a **large language model** to generate a concise answer.

Here’s a complete Python example using `psycopg2` for Postgres and OpenAI’s API:

```python
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
```

---

### Key Points in This Code

1. **Full-text search in Postgres**:

   * `to_tsvector` and `plainto_tsquery` allow you to search for relevant pages by keywords.
   * `ts_rank_cd` ranks results by relevance.

2. **RAG (Retrieval-Augmented Generation)**:

   * Only the **top_k most relevant pages** are fed to the LLM, reducing token usage.

3. **Prompt design**:

   * Instructs the LLM to **only answer using the provided text**.
   * Emphasizes **concise, numeric, and precise answers**.
   * Positions the LLM as a **50+ year expert analyst**, so answers are professional and insightful.

---

If you want, I can **also write an improved version** that:

* Splits large annual reports into smaller chunks (if a single page exceeds token limits).
* Generates **table-based answers** (like Revenue, Profit, Margin) automatically.

Do you want me to do that next?
