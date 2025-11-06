# train_small_expert.py
import argparse, json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

def build_prompt(instruction, context, output_json):
    sys = "You are a small expert model that outputs STRICT JSON only."
    user = f"Query:\n{instruction}\n\nRelevant snippets:\n{context}\n\nReturn ONLY the JSON object."
    assistant = json.dumps(output_json, ensure_ascii=False)
    return f"<s>[SYS]{sys}[/SYS]\n[INST]{user}[/INST]\n{assistant}</s>"

def tokenize(example, tokenizer, max_len=2048):
    prompt = build_prompt(example["instruction"], example.get("input", ""), example["output_json"])
    toks = tokenizer(prompt, truncation=True, max_length=max_len)
    toks["labels"] = toks["input_ids"].copy()
    return toks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="e.g. google/flan-t5-base or small LLaMA")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default="checkpoints/small_expert_lora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files=args.train_file, split="train")
    ds = ds.map(lambda ex: tokenize(ex, tokenizer, args.max_len), remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args_train = TrainingArguments(
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

    trainer = Trainer(model=model, args=args_train, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ LoRA fine-tuned small expert saved to", args.output_dir)

if __name__ == "__main__":
    main()
