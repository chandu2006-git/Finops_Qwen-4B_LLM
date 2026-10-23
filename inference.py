"""FinOps LLM inference using Qwen3-4B and the published LoRA adapter.

Model repository:
https://huggingface.co/Chandu06/finops-qwen3-4b

This module runs inference locally in a compatible GPU environment.
It does not create a public API endpoint by itself.
"""

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_MODEL = "Chandu06/finops-qwen3-4b"

SYSTEM_PROMPT = """You are a FinOps analysis assistant.

Analyze only the evidence provided. Distinguish facts from interpretations.
Perform calculations supported by the supplied values. Never invent pricing,
configuration, workload behavior, or historical data. Identify missing evidence
when necessary.

You exclusively assist with FinOps and cloud cost optimization. For unrelated
questions, respond:
"I can only assist with FinOps and cloud cost optimization analysis."

Use this structure:
OBSERVATIONS:
CALCULATIONS:
EVIDENCE:
ASSESSMENT:
RECOMMENDATION:
SAVINGS:
MISSING_EVIDENCE:
CONFIDENCE:
"""


def load_model():
    """Load the Qwen base model and attach the FinOps LoRA adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. This 4-bit inference configuration "
            "requires a compatible GPU runtime, such as the tested Kaggle setup."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    model.eval()

    return tokenizer, model


def generate_answer(question, tokenizer, model, max_new_tokens=512):
    """Generate a FinOps report for one user question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    tokenizer, model = load_model()
    question = input("Enter a FinOps question: ")
    print("\n--- FinOps LLM Response ---\n")
    print(generate_answer(question, tokenizer, model))
