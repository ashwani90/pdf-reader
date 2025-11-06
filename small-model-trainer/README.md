# Two-Stage QA: Small Expert (LoRA) + Base LLaMA

This project implements a **two-stage answering system**:
1) **Small Expert Model (LoRA-finetuned)** on your data that can either:
   - **(A)** answer directly, or
   - **(B)** produce a compact, factual context (short summary / set of facts) for the query.
2) **Base LLaMA model** that receives the user query **plus** the small-model output in its prompt and generates the final answer.

# Train the model

```

python train_small_expert.py \
  --base_model <your-small-instruct-model> \
  --train_file data/sample_train.jsonl \
  --output_dir checkpoints/small_expert_lora \
  --batch_size 2 --epochs 3 --lr 2e-4
```

# run the model
```
python serve_pipeline.py \
  --small_base_model <your-small-instruct-model> \
  --small_lora_path checkpoints/small_expert_lora \
  --base_llama meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

```