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
