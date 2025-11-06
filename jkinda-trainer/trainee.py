import os, json, random
from typing import Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import set_seed
from pathlib import Path
import transformers
print("TrainingArguments class from:", TrainingArguments.__module__)
print("Class file path:", TrainingArguments.__init__.__code__.co_filename)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
)

print("✅ TrainingArguments works fine")

# ========================
# Config
# ========================
MODEL_NAME = "distilgpt2"  # small, fast. You can try "gpt2" if you have more VRAM
DATA_PATH = "news.jsonl"   # your prepared jsonl with {"text": "..."}
OUTPUT_DIR = "news-autocomplete-model"
MAX_SEQ_LEN = 512          # 256-1024 depending on VRAM; 512 is a good start
SEED = 42
BATCH_SIZE = 4             # effective batch size = per_device_train_batch_size * gradient_accumulation_steps
GRAD_ACCUM = 8
EPOCHS = 3
LR = 5e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
EVAL_STEPS = 500
SAVE_STEPS = 1000

set_seed(SEED)

# ========================
# Load & prepare dataset
# ========================
assert Path(DATA_PATH).exists(), f"Missing {DATA_PATH}"

raw = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        txt = obj.get("text", "").strip()
        if txt:
            raw.append({"text": txt})

# train/val split (90/10)
random.shuffle(raw)
split = int(0.9 * len(raw))
train_list = raw[:split]
val_list = raw[split:]

train_ds = Dataset.from_list(train_list)
val_ds = Dataset.from_list(val_list)

# ========================
# Tokenizer & Model
# ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT2 family has no pad token; set it to eos for convenience
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ========================
# Tokenization (causal LM)
# ========================
def tok_fn(batch: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_attention_mask=True,
    )

tokenized_train = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
tokenized_val = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

# Data collator: creates labels by shifting inputs for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # IMPORTANT: causal LM, not masked LM
)
print("Transformers version in runtime:", transformers.__version__)
# ========================
# Training args & Trainer
# ========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    evaluation_strategy="steps",     # ✅ now works
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=100,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

def compute_metrics(eval_pred):
    # Report perplexity
    import math
    loss = float(eval_pred.metrics["eval_loss"]) if isinstance(eval_pred, dict) else None
    if loss is None:
        return {}
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = float("inf")
    return {"perplexity": ppl}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ========================
# Train
# ========================
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ========================
# Quick eval (perplexity)
# ========================
metrics = trainer.evaluate()
try:
    import math
    metrics["perplexity"] = math.exp(metrics["eval_loss"])
except Exception:
    metrics["perplexity"] = None
print("Eval metrics:", metrics)

# ========================
# Inference (autocomplete demo)
# ========================
prompt = "The central bank signaled"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
gen_ids = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.8,       # 0.7–1.0 gives variety
    top_p=0.95,            # nucleus sampling
    do_sample=True,
    repetition_penalty=1.1 # helps avoid loops
)
print("PROMPT:", prompt)
print("COMPLETION:", tokenizer.decode(gen_ids[0], skip_special_tokens=True))
