'''
Read the txt file name tt.txt
loop through content between delimiters (---||---)
 - get content between the delimiters
 - run prompt on that content (handle long content as well) (split if long content based on char limit (more than 5000 chars))
 - append the prompt output result to the report file
loop ends

save the report file
'''

"""
This is good but lets also have a set of prompts or questions that a analyst asks from the annual report

Using RAG we could implement that
"""

import os
from meta_ai_api import MetaAI
import json
import re
import time
import shutil
import psycopg2
from datetime import datetime

# Initialize client (ensure OPENAI_API_KEY is set in environment)

# Finance extraction prompt (short version with missing details handling)
# BASE_PROMPT = """
# You are a financial analyst AI. From the given company report text, extract and summarize available financial details.
# Include core metrics (revenue, profit, EPS, margins, cash flow, debt, assets/liabilities), growth trends, segment performance,
# major strategic moves, and risks/red flags. If some details are missing, note them as “Not mentioned.”
# Return the result as "JSON" and no other string just {
#     <key1> : <value1> 
#     <key2>: <value2>
#     ...
# } 
# Include things like Financial Highlights, Growth & Trends, Risks & Red Flags, Strategic Notes, Analyst Observations.
# """

BASE_PROMPT = """
Extract only the information a skilled financial analyst would find useful from the given financial text. Ignore generic commentary or non-financial content. Return clean, valid JSON with dynamic keys reflecting extracted data. Include only relevant quantitative and qualitative insights.

Possible keys (add or omit as needed): company_name, sector, fiscal_year, financial_metrics, growth_rates, assets, liabilities, debt, equity, valuation_metrics, market_trends, risk_factors, analyst_sentiment, data_source

Output example:

{
  "company_name": "ABC Ltd.",
  "fiscal_year": "2024-25",
  "financial_metrics": {
    "revenue_growth": "15.2%",
    "retained_earnings": "₹36T",
    "long_term_debt": "₹33.3T"
  },
  "valuation_metrics": {"debt_to_equity": "0.9"},
  "market_trends": "Expansion in non-current assets",
  "risk_factors": "Stretched valuations, weak demand"
}
Focus only on financially material data and omit null or irrelevant fields.
"""

# File paths
# only this needs to be changed
INPUT_FILE = "output/content/tata-motor/tata-motor1.txt"
OUTPUT_FILE = "output/report.jsonl"



# Helper: split text into chunks if too long
def split_text(text, max_chars=5000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def query_meta_llm(prompt):
    ne_title = False
    ne_content = False
    ai = MetaAI()
    try:
        response = ai.prompt(message=prompt)
        # logger.debug('Response from meta ai ' + str(response))
        if response:
            response = response.get("message")
            response = extract_json(response)
            return response
            # response = json.loads(response)
            # ne_title = response.get('title')
            # ne_content = response.get('description')
    except Exception as e:
        # logger.error('Exception in prompt processing ' + prompt)
        # logger.error(e)
        print(e)
        
    
    return response

def find_txt_files(base_dir):
    """
    Recursively find all .txt files in a directory and print their full paths.

    Args:
        base_dir (str): The starting directory to search.
    """
    new_path = False
    full_path = False
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                full_path = os.path.join(root, file)
                new_path = transform_path(full_path)
                
                break
    return new_path, full_path
def transform_path(path: str) -> str:
    """
    Replace 'content' with 'processed' in a path,
    and change extension from .txt to .json.
    """
    # Replace 'content' with 'processed'
    new_path = path.replace("content", "reports", 1)

    # Replace extension with .json
    base, _ = os.path.splitext(new_path)
    new_path = base + ".json"
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Create the file (empty if nothing written yet)
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("")  # or json.dump(data, f) if you have content

    return new_path

def extract_json(text: str):
    """
    Extracts the first valid JSON object or array from a given string.
    
    Args:
        text (str): Input string containing JSON + other text.
        
    Returns:
        dict/list: Parsed JSON if found.
        None: If no valid JSON is found.
    """
    # Regex to find potential JSON substrings (objects or arrays)
    candidates = re.findall(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

def remove_number_from_filename(file_path):
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    # Remove trailing digits from filename
    new_name = re.sub(r'\d+$', '', name)

    # Reconstruct full path
    new_path = os.path.join(dir_name, new_name + ext)
    return new_path

def process_section(section_text, new_path):
    results = []
    chunks = split_text(section_text)

    for idx, chunk in enumerate(chunks, start=1):
        # Combine prompt with chunk
        prompt = f"{BASE_PROMPT}\n\nCompany Report Excerpt (Part {idx}):\n{chunk}"

        # Call OpenAI API (chat completion)
        response = query_meta_llm(prompt)
        new_path = remove_number_from_filename(new_path)
        with open(new_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(response) + "\n")
        time.sleep(10)
        # results.append(response)

    return "\n\n".join(results)

def extract_company_name(filename):
    """
    Extracts a clean company name from filename or path.
    Removes numeric suffixes like 'tata-motor1' → 'tata-motor'.
    """
    # Normalize slashes for all OS
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)

    # Remove numeric suffix at end (e.g., motor1 -> motor)
    name = re.sub(r"\d+$", "", name)

    # Clean up hyphens/underscores and format nicely
    company_name = name.replace("_", "-").strip().lower()

    return company_name

def insert_prompt_record(cur, prompt, company_name):
    """
    Insert a new prompt record in rag_generated_prompts with status 'pending'
    """
    cur.execute("""
        INSERT INTO rag_generated_prompts (prompt, status, created_at, updated_at, company_name)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """, (prompt, 'pending', datetime.now(), datetime.now(), company_name))

    record_id = cur.fetchone()[0]
    print(f"📝 New prompt record inserted with ID {record_id}")
    return record_id

def main_fun_call():
    conn = psycopg2.connect(
        dbname="jkinda_stocks",
        user="ashwani_ocr",
        password="ashwani",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
    new_path, full_path = find_txt_files("output/content/")
    
    if not full_path:
        return False
    
    if not os.path.exists(full_path):
        print(f"Error: {full_path} not found")
        return False

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split sections by delimiter
    sections = content.split("---||---")
    for i, section in enumerate(sections, start=1):
        section = section.strip()
        if not section:
            continue

        print(f"Processing section {i}...")
        # process_section(section, new_path)
        prompt = f"{BASE_PROMPT}\n\nCompany Report Excerpt (Part {i}):\n{section}"
        company_name = extract_company_name(full_path)
        insert_prompt_record(cur, prompt, company_name)
        conn.commit()
        # report.write(f"\n\n=== Report for Section {i} ===\n")
        # report.write(summary)
        # report.write("\n\n")

    print(f"Report saved to {new_path}")
    move_path = full_path.replace("content", "processed", 1)
    os.makedirs(os.path.dirname(move_path), exist_ok=True)

    # Move the file
    shutil.move(full_path, move_path)
    
def main():
    val = True
    while val:
        val = main_fun_call()
    print("All done.")

if __name__ == "__main__":
    main()
