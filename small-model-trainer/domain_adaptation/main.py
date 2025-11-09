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
