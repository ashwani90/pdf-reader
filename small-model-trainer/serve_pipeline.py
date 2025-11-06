# serve_pipeline.py
import os, json, argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from jsonschema import validate

from prompts import (
    SMALL_EXPERT_SYSTEM_PROMPT, SMALL_EXPERT_USER_TEMPLATE,
    BASE_LLAMA_SYSTEM_PROMPT, BASE_LLAMA_USER_TEMPLATE
)

SMALL_JSON_SCHEMA = {
  "type": "object",
  "properties": {
    "mode": {"type": "string", "enum": ["answer", "facts"]},
    "answer": {"type": "string"},
    "facts": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["mode", "answer", "facts"]
}

class QARequest(BaseModel):
    query: str
    extras: dict | None = None

def format_small_prompt(query, context):
    sys = SMALL_EXPERT_SYSTEM_PROMPT
    user = SMALL_EXPERT_USER_TEMPLATE.format(query=query, context=context or "(no snippets)")
    return f"<s>[SYS]{sys}[/SYS]\n[INST]{user}[/INST]"

def format_base_prompt(query, small_json_text):
    sys = BASE_LLAMA_SYSTEM_PROMPT
    user = BASE_LLAMA_USER_TEMPLATE.format(query=query, small_json=small_json_text)
    return f"<s>[SYS]{sys}[/SYS]\n[INST]{user}[/INST]"

def load_small_expert(base_model, lora_path):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(base_model)
    mdl = PeftModel.from_pretrained(mdl, lora_path)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
    return gen, tok

def load_base_llama(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    gen = pipeline("text-generation", model=model_name, tokenizer=tok, device_map="auto")
    return gen, tok

def generate_json(gen, prompt, max_new_tokens=256):
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    start, end = out.find("{"), out.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Small expert did not return JSON.")
    return out[start:end+1]

def route_and_answer(small_obj):
    ans = None
    route = "base"
    if small_obj.get("mode") == "answer":
        txt = (small_obj.get("answer") or "").strip()
        if 0 < len(txt) <= 480:
            ans = txt
            route = "direct"
    return route, ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_base_model", required=True)
    parser.add_argument("--small_lora_path", required=True)
    parser.add_argument("--base_llama", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    small_gen, _ = load_small_expert(args.small_base_model, args.small_lora_path)
    base_gen, _ = load_base_llama(args.base_llama)

    app = FastAPI()

    @app.post("/qa")
    def qa(req: QARequest):
        # TODO: Plug your retriever to fill 'context'
        context = ""
        sprompt = format_small_prompt(req.query, context)
        try:
            raw_json = generate_json(small_gen, sprompt)
            obj = json.loads(raw_json)
            validate(instance=obj, schema=SMALL_JSON_SCHEMA)
        except Exception:
            obj = {"mode": "facts", "facts": [], "answer": ""}

        route, direct = route_and_answer(obj)
        if route == "direct" and direct:
            return {"route": "direct", "answer": direct, "small_json": obj}

        # Combine with base model
        small_text = json.dumps(obj, ensure_ascii=False)
        bprompt = format_base_prompt(req.query, small_text)
        final = base_gen(bprompt, max_new_tokens=384, do_sample=False)[0]["generated_text"]
        final = final.split("[/INST]")[-1].strip()
        return {"route": "base", "answer": final, "small_json": obj}

    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
