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
import json
import re

# ========= CONFIG ========= #
DB_NAME = "jkinda_stocks"
DB_USER = "ashwani_ocr"
DB_PASSWORD = "ashwani"
DB_HOST = "localhost"
DB_PORT = "5432"

TABLE_NAME = "rag_generated_prompts"
OUTPUT_DIR = "output/answers"
# ========================== #

def merge_json_objects(json_list, separator="--|--"):
    def merge_values(existing, new):
        # If both are dicts → merge recursively
        if isinstance(existing, dict) and isinstance(new, dict):
            return merge_dicts(existing, new)

        # Convert both values to string for merging
        return f"{existing}{separator}{new}"

    def merge_dicts(d1, d2):
        merged = dict(d1)
        for key, val in d2.items():
            if key in merged:
                merged[key] = merge_values(merged[key], val)
            else:
                merged[key] = val
        return merged

    result = {}
    for obj in json_list:
        result = merge_dicts(result, obj)

    return result

def extract_and_fix_json(raw_text: str):
    """
    Extract JSON block from messy LLM response text and auto-fix common formatting errors.
    Returns a Python dict or None if extraction fails.
    """

    # ✅ Step 1: Try to isolate the JSON using regex
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        print("❌ No JSON found in text")
        return None

    json_text = match.group(0)

    # ✅ Step 2: Auto-clean common LLM formatting mistakes
    fixes = [
        (r",\s*}", "}"),             # Remove trailing commas before closing brace
        (r",\s*]", "]"),             # Remove trailing commas before closing bracket
        (r"[\u201C\u201D]", '"'),    # Replace fancy quotes with standard quotes
    ]
    for pattern, repl in fixes:
        json_text = re.sub(pattern, repl, json_text)

    # ✅ Step 3: Try parsing, retry with additional fixes if needed
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print("⚠ JSON decode error, trying fallback fix:", e)

        # Second pass — sometimes quotes missing on keys
        json_text_fixed = re.sub(r"(\w+):", r'"\1":', json_text)
        
        try:
            return json.loads(json_text_fixed)
        except Exception as e2:
            print("❌ Failed to decode JSON:", e2)
            print("📌 Final extracted attempt:\n", json_text_fixed)
            return None

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
        WHERE company_name = %s AND status='answered'
        ORDER BY created_at ASC;
    """
    cur.execute(query, (company_name,))
    rows = cur.fetchall()
    json_lists = []
    for row in rows:
        
        try:
            abc = json.loads(row[2])
            
            abc = extract_and_fix_json(abc['message'])
            json_lists.append(abc)
        except Exception as e:
            print("Error parsing JSON for record ID", row[0], ":", e)
    merged_json = merge_json_objects(json_lists)
    print(merged_json)
    cur.close()
    conn.close()

    return merged_json

# TODO:: Analysis to be done later on
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
