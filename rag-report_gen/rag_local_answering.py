"""
RAG: retrieve relevant excerpts from Postgres (pgvector) and answer questions using a local open-source LLM.

Requires:
- psycopg2-binary
- sentence-transformers
- pgvector
- transformers (or llama-cpp-python if using that backend)
- torch / accelerate (for HF)
"""

import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from datetime import datetime
# -------------------------
# CONFIG
# -------------------------
DB_CONFIG = {
    "dbname": "jkinda_stocks",
    "user": "ashwani_ocr",
    "password": "ashwani",
    "host": "localhost",
    "port": 5432,
}

# Names of tables/columns
QUESTIONS_TABLE = "rag_questions"           # holds questions to answer
QUESTIONS_ID_COL = "id"
QUESTIONS_TEXT_COL = "question"
QUESTIONS_ANSWER_COL = "answer"             # update with generated answer

EXCERPTS_TABLE = "financial_excerpts"      # retrieval collection
EXCERPTS_ID_COL = "id"
EXCERPTS_TEXT_COL = "excerpt"
EXCERPTS_FILENAME_COL = "filename"
# This needs to be changed
COMPANY = "Lg-el"                     # filter excerpts by filename starting with this company name
OUTPUT_FILE = "output/processed/Lg-el/Lg-el.txt"
# EXCERPTS embedding column assumed to be "embedding" of type vector(d)

TOP_K = 5            # number of top passages to retrieve per question
EMBED_MODEL_NAME = "intfloat/e5-large-v2"  # same model used before for embeddings

# LLM options: "hf" for transformers pipeline OR "llama_cpp" for llama-cpp-python
MODEL_CHOICE = "hf"   # or "llama_cpp"

# If using HF transformers, set the local model path/name (must be available locally or from HF)
HF_MODEL = "your-hf-local-or-hub-model"  # e.g., "meta-llama/Llama-2-7b-chat-hf" (you must have access and hardware)

# If using llama_cpp, set path to llama.cpp .bin file
LLAMA_CPP_MODEL_PATH = "/path/to/ggml-model-q4_0.bin"

# Generation params
MAX_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 0.95

def insert_prompt_record(cur, prompt, company_name, conn):
    """
    Insert a new prompt record in rag_generated_prompts with status 'pending'
    """
    try:
        cur.execute("""
            INSERT INTO rag_generated_prompts (prompt, status, created_at, updated_at, company_name,"type")
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (prompt, 'pending', datetime.now(), datetime.now(), company_name, "1"))

        # record_id = cur.fetchone()[0]
        # print(f"📝 New prompt record inserted with ID {record_id}")
    except Exception as e:
        print(e)
        # record_id = None
    # return record_id
    conn.commit()


def save_prompt_to_file(question_id, question_text, passages, prompt):
    """
    Save the generated RAG prompt and context passages to a file.
    Each record is separated by a line of 100 hyphens.
    """
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # f.write(f"Question ID: {question_id}\n")
        # f.write(f"Question: {question_text}\n\n")

        # f.write("Top Retrieved Passages:\n")
        # for idx, (pid, excerpt, filename) in enumerate(passages, 1):
        #     f.write(f"  Passage {idx} (File: {filename}, ID: {pid}):\n")
        #     f.write(excerpt.strip() + "\n\n")

        # f.write("Generated Prompt:\n")
        f.write(prompt.strip() + "\n")

        f.write("\n" + ("-" * 100) + "\n\n")

    print(f"✅ Prompt for Question {question_id} saved to {OUTPUT_FILE}")

# -------------------------
# Helpers
# -------------------------
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    # register pgvector adapter so we can pass Python lists directly as vectors
    register_vector(conn)
    return conn

def build_prompt(question_text, retrieved_passages):
    """
    Build a prompt for the LLM by placing the retrieved passages as context, then the question.
    Keep it concise: we want LLM to answer using the context and say "I don't know" if not found.
    """
    context = "\n\n---\n\n".join(
        [f"Source ({p[EXCERPTS_FILENAME_COL]} id={p[EXCERPTS_ID_COL]}):\n{p[EXCERPTS_TEXT_COL]}" for p in retrieved_passages]
    )

    prompt = (
        "You are an expert assistant specialized in financial reports.\n"
        "Use ONLY the provided context passages to answer the question. If the answer is not present, say you cannot find it.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Answer (concise, reference sources by id if useful):"
    )
    return prompt

# -------------------------
# Embedding model
# -------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# -------------------------
# LLM setup (HF or llama_cpp)
# -------------------------
# llm = None
# if MODEL_CHOICE == "hf":
#     # Hugging Face transformers pipeline (requires a local HF-compatible model)
#     from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#     print("Loading HF model/tokenizer... (this may take a while)")
#     tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map="auto", torch_dtype="auto")
#     text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map='auto', 
#                         max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
#     def llm_generate(prompt):
#         # HF pipeline returns list of dicts
#         out = text_gen(prompt, max_new_tokens=MAX_TOKENS, do_sample=(TEMPERATURE>0))
#         return out[0]["generated_text"][len(prompt):].strip()
# elif MODEL_CHOICE == "llama_cpp":
#     # llama-cpp-python
#     from llama_cpp import Llama
#     print("Loading llama-cpp model...")
#     llm = Llama(model_path=LLAMA_CPP_MODEL_PATH)
#     def llm_generate(prompt):
#         resp = llm.create(prompt=prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
#         return resp["choices"][0]["text"].strip()
# else:
#     raise ValueError("Unsupported MODEL_CHOICE. Choose 'hf' or 'llama_cpp'.")

# -------------------------
# Main RAG loop
# -------------------------
def main():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Fetch questions that are unanswered (embedding or answer is null)
    # Adjust condition to your schema. Here we choose rows where answer is NULL.
    cur.execute(f"SELECT {QUESTIONS_ID_COL}, {QUESTIONS_TEXT_COL} FROM {QUESTIONS_TABLE};")
    questions = cur.fetchall()
    print(f"Found {len(questions)} unanswered questions.")

    for q in questions:
        qid = q[QUESTIONS_ID_COL]
        qtext = q[QUESTIONS_TEXT_COL]
        print(f"\nProcessing question id={qid}: {qtext}")

        # 1) Embed the question
        q_emb = embed_model.encode(f"query: {qtext}").tolist()

        # 2) Retrieve top-k passages using pgvector '<->' operator
        vector_str = "[" + ",".join(map(str, q_emb)) + "]"
        
        cur.execute(
            f"""
            SELECT {EXCERPTS_ID_COL}, {EXCERPTS_TEXT_COL}, {EXCERPTS_FILENAME_COL}
            FROM {EXCERPTS_TABLE}
            WHERE filename LIKE %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """,
                (f"{COMPANY}%", vector_str, TOP_K)
            )
        passages = cur.fetchall()
        print(f" Retrieved {len(passages)} passages.")

        if not passages:
            ans_text = "I could not find relevant information in the documents."
        else:
            # 3) Build prompt with retrieved passages
            prompt = build_prompt(qtext, passages)
            insert_prompt_record(cur, prompt, COMPANY, conn)
            # print(passages)
            # save_prompt_to_file(qid, qtext, passages, prompt)
            # print(prompt)

            # 4) Ask LLM to generate answer from context
            
            # try:
            #     ans_text = llm_generate(prompt)
            # except Exception as e:
            #     print("LLM generation error:", e)
            #     ans_text = "Error generating answer."

        # 5) Save the answer back to DB
        # cur.execute(
        #     f"UPDATE {QUESTIONS_TABLE} SET {QUESTIONS_ANSWER_COL} = %s WHERE {QUESTIONS_ID_COL} = %s;",
        #     (ans_text, qid)
        # )
        # conn.commit()
        print(f" Saved answer for question id={qid}.")

    cur.close()
    conn.close()
    print("All done.")

if __name__ == "__main__":
    main()
