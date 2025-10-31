#!/usr/bin/env python3
"""
Script: export_company_answers.py
Description:
    Reads the `rag_generated_prompts` table and collects all `answer`
    values filtered by `company_name`.

Usage:
    python export_company_answers.py tata-motor
"""

import psycopg2
import sys
from datetime import datetime

# ========= CONFIG ========= #
DB_NAME = "jkinda_stocks"
DB_USER = "ashwani_ocr"
DB_PASSWORD = "ashwani"
DB_HOST = "localhost"
DB_PORT = "5432"

TABLE_NAME = "rag_generated_prompts"
OUTPUT_DIR = "output/answers"
# ========================== #

def get_company_answers(company_name):
    """
    Fetch all answers for a given company_name from rag_generated_prompts table.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    query = f"""
        SELECT id, prompt, answer, status, created_at
        FROM {TABLE_NAME}
        WHERE company_name = %s
        ORDER BY created_at ASC;
    """
    cur.execute(query, (company_name,))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


# def save_answers_to_file(company_name, rows):
#     """
#     Save all answers to a timestamped output file.
#     """
#     import os
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{OUTPUT_DIR}/{company_name}_answers_{timestamp}.txt"

#     with open(filename, "w", encoding="utf-8") as f:
#         for (rid, prompt, answer, status, created_at) in rows:
#             f.write(f"### Record ID: {rid}\n")
#             f.write(f"Created At: {created_at}\n")
#             f.write(f"Status: {status}\n\n")
#             f.write(f"Prompt:\n{prompt}\n\n")
#             f.write(f"Answer:\n{answer or '[No Answer]'}\n")
#             f.write("\n" + "-" * 100 + "\n\n")

#     print(f"✅ Exported {len(rows)} records to {filename}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_company_answers.py <company_name>")
        sys.exit(1)

    company_name = sys.argv[1].lower()
    rows = get_company_answers(company_name)

    if not rows:
        print(f"No records found for company: {company_name}")
        return

    print(f"Found {len(rows)} records for '{company_name}'. Exporting...")


if __name__ == "__main__":
    main()
