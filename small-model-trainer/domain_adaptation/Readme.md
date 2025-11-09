We’ll use **FLAN-T5-small** with **LoRA adapters**, so it fits easily on your 4GB GPU.
You can run both stages sequentially without retraining from scratch.

---

# 🧩 Stage 1: Domain Adaptation — Teach the Model “News Language”

This trains the model to **reconstruct its input text**, which is how we adapt it to a new domain without labels.

---

```python
# ============================
# domain_adaptation.py
# ============================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/flan-t5-small",
                        help="Base model name or path")
    parser.add_argument("--train_file", required=True,
                        help="Path to your text corpus (e.g., data/news.txt)")
    parser.add_argument("--output_dir", default="checkpoints/domain_adapted",
                        help="Where to save domain-adapted LoRA weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    # ---------------------------
    # Load the raw text dataset
    # ---------------------------
    # Hugging Face will create a Dataset object from your text file.
    # Each entry will be {'text': "your paragraph here"}
    ds = load_dataset("text", data_files=args.train_file, split="train")

    # ---------------------------
    # Load tokenizer and base model
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # ---------------------------
    # Configure LoRA adapters
    # ---------------------------
    # LoRA (Low-Rank Adaptation) inserts small trainable matrices
    # into certain layers of the model (like attention projections)
    # so we can fine-tune efficiently without touching all weights.
    lora_cfg = LoraConfig(
        r=16,                     # rank of adapter matrices (smaller = lighter)
        lora_alpha=32,            # scaling factor for LoRA layers
        lora_dropout=0.05,        # dropout for regularization
        bias="none",
        task_type="SEQ_2_SEQ_LM"  # because FLAN-T5 is encoder-decoder
    )
    model = get_peft_model(model, lora_cfg)

    # ---------------------------
    # Tokenization function
    # ---------------------------
    # Here, we train the model to "reconstruct its own text"
    # (labels = input_ids). This is self-supervised learning
    # similar to pretraining.
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Data collator handles dynamic padding and batching
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ---------------------------
    # Training arguments
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # accumulates gradients to simulate larger batch
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[]
    )

    # ---------------------------
    # Trainer setup
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator
    )

    # ---------------------------
    # Train the domain-adapted model
    # ---------------------------
    trainer.train()

    # ---------------------------
    # Save the adapted model (LoRA weights)
    # ---------------------------
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Domain-adapted model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

---

# 🚀 Stage 2: Task Fine-Tuning — Teach the Model to Extract Facts

Now that the model understands “news language,”
we fine-tune it on labeled examples where you specify **input → output JSON** pairs.

---

```python
# ============================
# task_finetune.py
# ============================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/flan-t5-small",
                        help="Base model (should be same as Stage 1)")
    parser.add_argument("--domain_lora_path", default="checkpoints/domain_adapted",
                        help="Path to domain-adapted LoRA weights from Stage 1")
    parser.add_argument("--train_file", required=True,
                        help="Path to your instruction dataset (JSONL)")
    parser.add_argument("--output_dir", default="checkpoints/fact_extractor_lora",
                        help="Where to save task-finetuned model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    # ---------------------------
    # Load dataset (instruction format)
    # ---------------------------
    # Example JSONL line:
    # {"input": "Text about Tesla...", "output": "{\"facts\": [\"Tesla opened factory\"]}"}
    ds = load_dataset("json", data_files=args.train_file, split="train")

    # ---------------------------
    # Load tokenizer and model
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # Load domain-adapted LoRA weights from Stage 1
    model = PeftModel.from_pretrained(base_model, args.domain_lora_path)

    # ---------------------------
    # Prepare prompts
    # ---------------------------
    # Each input becomes a clear instruction to the model
    def make_prompt(example):
        instruction = (
            "Extract key factual statements or a concise summary "
            "from the following text in JSON format."
        )
        combined = f"{instruction}\n\nText:\n{example['input']}\n\nOutput JSON:"
        return {"input_text": combined, "target_text": example["output"]}

    ds = ds.map(make_prompt)

    # ---------------------------
    # Tokenization
    # ---------------------------
    def tokenize(example):
        model_inputs = tokenizer(
            example["input_text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["target_text"],
                truncation=True,
                padding="max_length",
                max_length=args.max_len
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ---------------------------
    # Training setup
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[]
    )

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Task-finetuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

---

# 🔍 Explanation Summary

| Stage                           | Goal                                              | Input                     | Output                       |
| ------------------------------- | ------------------------------------------------- | ------------------------- | ---------------------------- |
| **Stage 1 — Domain Adaptation** | Teach model to understand domain text             | Raw news articles         | Domain-adapted LoRA weights  |
| **Stage 2 — Task Fine-Tuning**  | Teach model to extract facts in structured format | Instruction + text → JSON | Fact extraction LoRA weights |

---

# 🧱 Folder Structure Example

```
data/
 ├── news.txt                 ← raw text for domain adaptation
 ├── news_facts.jsonl         ← labeled data for fact extraction
checkpoints/
 ├── domain_adapted/          ← LoRA adapter from Stage 1
 └── fact_extractor_lora/     ← final fine-tuned adapter
```

---

# ⚡ How to Run

### Stage 1 — Domain Adaptation

```bash
python domain_adaptation.py \
  --train_file data/news.txt \
  --output_dir checkpoints/domain_adapted
```

### Stage 2 — Task Fine-tuning

```bash
python task_finetune.py \
  --domain_lora_path checkpoints/domain_adapted \
  --train_file data/news_facts.jsonl \
  --output_dir checkpoints/fact_extractor_lora
```

---

# 🧠 What You Get

After both stages:

* The model “speaks news” (understands domain vocabulary and context).
* It knows how to **extract structured facts** in JSON.
* It fits easily on a **4 GB GPU** using LoRA adapters.
* You can load and query it in your existing `serve_pipeline.py` like this:

```python
mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
mdl = PeftModel.from_pretrained(mdl, "checkpoints/fact_extractor_lora")
```

# Search Retriever

We’ll build a file called **`search_retriever.py`** that you can plug directly into your `serve_pipeline.py` later.

---

# 🧩 What This Script Does

✅ Reads your text corpus (news articles).
✅ Splits them into overlapping chunks (so each piece fits the model’s context).
✅ Uses a **SentenceTransformer** (embedding model) to convert each chunk into a vector.
✅ Stores all vectors in a **FAISS index** (on disk).
✅ Lets you query the index to retrieve the most semantically similar passages.
✅ Optionally, you can combine it with your LoRA-fine-tuned model to summarize or extract facts.

---

# ⚙️ Prerequisites

Before running, install:

```bash
pip install faiss-cpu sentence-transformers
```

(If you have GPU, use `faiss-gpu` instead of `faiss-cpu`.)

---

# 🧠 Explanation

| Section                                   | What It Does                                                               | Why It Matters                                       |
| ----------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------- |
| `build_faiss_index()`                     | Reads your text file, splits it, embeds, and builds FAISS index            | Converts unstructured text → searchable vector space |
| `search_faiss()`                          | Takes a natural-language query, embeds it, and searches for similar chunks | Lets you retrieve relevant text pieces               |
| `RecursiveCharacterTextSplitter`          | Breaks large text into 500-character chunks                                | Keeps context small for LLM input                    |
| `SentenceTransformer("all-MiniLM-L6-v2")` | Turns text into numeric embeddings                                         | Enables semantic search (not keyword search)         |
| `faiss.IndexFlatL2`                       | Vector index using L2 (Euclidean) distance                                 | Efficient similarity search                          |
| `chunks.pkl`                              | Stores your text pieces alongside the index                                | Needed for later retrieval                           |
| `top_k`                                   | Number of top results to retrieve                                          | Controls context breadth for RAG                     |

---

# ⚡ Example Usage

### 1️⃣ Build the index

```bash
python search_retriever.py --build --text_file data/news.txt --index_path faiss_index
```

Output:

```
📖 Reading corpus from data/news.txt...
✅ Split corpus into 1280 chunks.
🔹 Encoding chunks into embeddings...
✅ FAISS index built with 1280 vectors.
💾 Saved FAISS index and chunks to faiss_index/
```

---

### 2️⃣ Query the index

```bash
python search_retriever.py --query "Which company opened a factory in Texas?" --index_path faiss_index
```

Output:

```
🔹 Top 3 results for query: 'Which company opened a factory in Texas?'

1. (distance=0.42)
Tesla announced plans to open a new electric battery factory in Austin, Texas, in 2021...

2. (distance=0.56)
The Texas governor attended the opening ceremony of the new Tesla factory...
```

---

# 🧩 Step 3: Integrate with Your Fine-Tuned Model

Now, you can modify your existing `serve_pipeline.py` (in `/qa` endpoint):

```python
from search_retriever import search_faiss

@app.post("/qa")
def qa(req: QARequest):
    # Step 1: Retrieve relevant text from corpus
    retrieved_chunks = search_faiss(req.query, index_path="faiss_index", top_k=3)
    context = "\n\n".join(retrieved_chunks)

    # Step 2: Build prompt for small expert
    sprompt = format_small_prompt(req.query, context)

    # Step 3: Generate structured facts
    raw_json = generate_json(small_gen, sprompt)
    obj = json.loads(raw_json)
    ...
```

✅ This gives you a complete **RAG (Retrieval-Augmented Generation)** system:

* **Retriever** → finds relevant news chunks via FAISS.
* **Generator (your small model)** → summarizes or extracts structured facts.
* **Optional base LLaMA** → synthesizes or explains results.

---

# 🧠 TL;DR Summary

| Component              | Purpose                                               |
| ---------------------- | ----------------------------------------------------- |
| `domain_adaptation.py` | Make the model fluent in your domain language         |
| `task_finetune.py`     | Teach the model structured fact extraction            |
| `search_retriever.py`  | Provide fast semantic retrieval over your text corpus |
| `serve_pipeline.py`    | Combine retrieval + generation for interactive Q&A    |
---
