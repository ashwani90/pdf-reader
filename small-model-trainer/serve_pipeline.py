# serve_pipeline.py
import os, json, argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel
from jsonschema import validate

from prompts import (
    SMALL_EXPERT_SYSTEM_PROMPT, SMALL_EXPERT_USER_TEMPLATE,
    BASE_LLAMA_SYSTEM_PROMPT, BASE_LLAMA_USER_TEMPLATE
)

# Schema to validate the small expert's JSON output
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
    """Builds the small expert prompt (FLAN-T5 style)."""
    sys = SMALL_EXPERT_SYSTEM_PROMPT
    user = SMALL_EXPERT_USER_TEMPLATE.format(query=query, context=context or "(no snippets)")
    return f"{sys}\n\n{user}"


def format_base_prompt(query, small_json_text):
    """Builds the base LLaMA input prompt."""
    sys = BASE_LLAMA_SYSTEM_PROMPT
    user = BASE_LLAMA_USER_TEMPLATE.format(query=query, small_json=small_json_text)
    return f"<s>[SYS]{sys}[/SYS]\n[INST]{user}[/INST]"


def load_small_expert(base_model, lora_path):
    """
    Loads the LoRA fine-tuned small expert.
    Works for FLAN-T5 or other Seq2Seq models.
    """
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    mdl = PeftModel.from_pretrained(mdl, lora_path)
    # FLAN-T5 works with the text2text-generation pipeline
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device_map="auto")
    return gen, tok


def load_base_llama(model_name):
    """Loads the larger LLaMA-style model (CausalLM)."""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    gen = pipeline("text-generation", model=model_name, tokenizer=tok, device_map="auto")
    return gen, tok


def generate_json(gen, prompt, max_new_tokens=256):
    """Runs the small expert and extracts JSON from its text output."""
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    start, end = out.find("{"), out.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("❌ Small expert did not return valid JSON.")
    return out[start:end+1]


def route_and_answer(small_obj):
    """Simple routing heuristic."""
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
    parser.add_argument("--small_base_model", required=True,
                        help="Base name of the small expert (e.g. google/flan-t5-base)")
    parser.add_argument("--small_lora_path", required=True,
                        help="Path to the fine-tuned LoRA weights")
    # parser.add_argument("--base_llama", required=True,
    #                     help="Name of the larger model, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("🔹 Loading small expert...")
    small_gen, _ = load_small_expert(args.small_base_model, args.small_lora_path)
    print("🔹 Loading base LLaMA...")
    # base_gen, _ = load_base_llama(args.base_llama)

    app = FastAPI(title="Two-Stage QA System")

    @app.post("/qa")
    def qa(req: QARequest):
        # TODO: integrate retriever later
        context = ""
        sprompt = format_small_prompt(req.query, context)

        try:
            raw_json = generate_json(small_gen, sprompt)
            obj = json.loads(raw_json)
            validate(instance=obj, schema=SMALL_JSON_SCHEMA)
        except Exception as e:
            print("⚠️  Error parsing small expert output:", e)
            obj = {"mode": "facts", "facts": [], "answer": ""}

        route, direct = route_and_answer(obj)
        if route == "direct" and direct:
            return {"route": "direct", "answer": direct, "small_json": obj}

        # Route through base model
        small_text = json.dumps(obj, ensure_ascii=False)
        bprompt = format_base_prompt(req.query, small_text)
        # final = base_gen(bprompt, max_new_tokens=384, do_sample=False)[0]["generated_text"]
        # final = final.split("[/INST]")[-1].strip()

        return {"route": "base", "answer": bprompt, "small_json": obj}

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
