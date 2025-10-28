import json
import re

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

import os

def find_txt_files(base_dir):
    """
    Recursively find all .txt files in a directory and print their full paths.

    Args:
        base_dir (str): The starting directory to search.
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                full_path = os.path.join(root, file)
                transform_path(full_path)
                print(full_path)
                break
            
def transform_path(path: str) -> str:
    """
    Replace 'content' with 'processed' in a path,
    and change extension from .txt to .json.
    """
    # Replace 'content' with 'processed'
    new_path = path.replace("content", "processed", 1)

    # Replace extension with .json
    base, _ = os.path.splitext(new_path)
    new_path = base + ".json"
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Create the file (empty if nothing written yet)
    with open(new_path, "w", encoding="utf-8") as f:
        f.write("")  # or json.dump(data, f) if you have content

    return new_path


if __name__ == "__main__":
    tr = "Given the company report excerpt provided, it seems to focus more on statutory reports and disputes rather than financial performance metrics like revenue, profit, EPS, etc. Based on the information available:\n{\n    \"Financial Highlights\": {\n        \"Revenue\": \"Not mentioned\",\n        \"Profit\": \"Not mentioned\",\n        \"EPS\": \"Not mentioned\",\n        \"Margins\": \"Not mentioned\",\n        \"Cash Flow\": \"Not mentioned\",\n        \"Debt\": \"Not mentioned\",\n        \"Assets/Liabilities\": \"Not mentioned\"\n    },\n    \"Growth & Trends\": {\n        \"Revenue Growth\": \"Not mentioned\",\n        \"Profit Growth\": \"Not mentioned\",\n        \"Segment Growth Trends\": \"Not mentioned\"\n    },\n    \"Segment Performance\": \"Not mentioned\",\n    \"Major Strategic Moves\": \"Not mentioned\",\n    \"Risks & Red Flags\": {\n        \"Disputed Dues\": {\n            \"Sales Tax\": \"Rs 396.14 crores (High Court & State Tribunals & Appellate Authority)\",\n            \"Goods and Service Tax\": \"Rs 732.98 crores (High Court, The Goods and Services Tax Appellate Tribunal & Appellate Authority)\",\n            \"Customs Act Duty\": \"Rs 47.52 crores (High Court & The Custom, Excise and Service Tax Appellate Tribunal & Appellate Authority)\",\n            \"Octroi\": \"Rs 66.47 crores (High Court & Supreme Court)\",\n            \"Property Tax\": \"Rs 209.45 crores (High Court & Civil Judge Sr. Division, Pune)\",\n            \"ESI Contribution\": \"Rs 1.70 crores (ESI Court)\"\n        }\n    },\n    \"Strategic Notes\": \"The company seems to be dealing with various legal and statutory disputes, particularly related to taxes and contributions.\",\n    \"Analyst Observations\": \"Potential financial impact from these disputes could be significant. Investors should monitor the outcomes of these cases closely.\"\n}\n"
    find_txt_files("output/")