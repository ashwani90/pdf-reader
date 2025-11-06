# Two-Stage QA: Small Expert (LoRA) + Base LLaMA

This project implements a **two-stage answering system**:
1) **Small Expert Model (LoRA-finetuned)** on your data that can either:
   - **(A)** answer directly, or
   - **(B)** produce a compact, factual context (short summary / set of facts) for the query.
2) **Base LLaMA model** that receives the user query **plus** the small-model output in its prompt and generates the final answer.
