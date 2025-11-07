# train_small_expert.py
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True,
                        help="e.g. google/flan-t5-base or small LLaMA")
    parser.add_argument("--train_file", required=True,
                        help="Plain text file — one training example per line.")
    parser.add_argument("--output_dir", default="checkpoints/small_expert_lora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=1024)
    args = parser.parse_args()

    # -----------------------------
    # Load tokenizer and base model
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # LoRA configuration for Seq2Seq model
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_cfg)

    # -----------------------------
    # Prepare dataset from text file
    # -----------------------------
    # Each line of the text file is a training document.
    # We'll turn each into a summarization-style prompt.
    ds = load_dataset("text", data_files=args.train_file, split="train")

    def make_prompt(example):
        text = example["text"].strip()
        # Prompt style for instruction fine-tuning
        # You can change this to a custom instruction style if you like
        instruction = (
            "Extract key factual statements or a concise summary "
            "from the following text in JSON format."
        )
        combined = f"{instruction}\n\nText:\n{text}\n\nOutput JSON:"
        return {"input_text": combined, "target_text": ""}

    ds = ds.map(make_prompt)

    def tokenize(ex):
        tok = tokenizer(
            ex["input_text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len
        )
        # For Seq2Seq, we can train it to reconstruct the same text or an empty target
        with tokenizer.as_target_tokenizer():
            target = tokenizer(
                ex["target_text"],
                truncation=True,
                padding="max_length",
                max_length=args.max_len
            )
        tok["labels"] = target["input_ids"]
        return tok

    ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # -----------------------------
    # Training setup
    # -----------------------------
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

    trainer = Trainer(model=model,
                      args=args_train,
                      train_dataset=ds,
                      data_collator=collator)

    # -----------------------------
    # Train and save
    # -----------------------------
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ LoRA fine-tuned small expert saved to", args.output_dir)


if __name__ == "__main__":
    main()
