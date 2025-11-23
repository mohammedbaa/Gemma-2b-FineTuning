📚 Fine-Tuning Gemma-2B for Quote Generation

A lightweight QLoRA fine-tuning project for generating quotes + their authors

This repository demonstrates how to fine-tune Google’s Gemma-2B Large Language Model using QLoRA on a consumer GPU (e.g., Google Colab T4).
The model is trained on the Abirate/english_quotes dataset to memorize quotes and correctly attribute them to their authors.

🚀 Project Overview

Base Model: google/gemma-2b

Technique: QLoRA (4-bit quantization + LoRA adapters)

Frameworks: Hugging Face trl, transformers, peft, bitsandbytes

Goal:
Input → partial quote
Output → completed quote + correct author name

🛠️ Installation & Requirements

Install the required packages exactly as used in this project:

pip install -q --upgrade torch
pip install -q --upgrade transformers==4.38.2
pip install -q --upgrade trl==0.7.10
pip install -q --upgrade peft==0.8.2
pip install -q --upgrade bitsandbytes==0.42.0
pip install -q --upgrade accelerate==0.27.2

📂 Dataset

The model is trained on the English Quotes Dataset:
👉 https://huggingface.co/datasets/abirate/english_quotes

Each training sample is pre-processed into the following strict text format:

Quote: {quote_text}
Author: {author_name}


This simple input-output format helps the model efficiently learn quote continuation and attribution.

⚙️ Training Configuration

This project uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA, allowing training on small GPUs.

Parameter	Value
LoRA Rank (r)	8
Target Modules	q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj
Quantization	4-bit NF4
Optimizer	paged_adamw_8bit
Training Steps	100

QLoRA reduces memory usage significantly, enabling training on free Colab GPUs.

💻 Model Loading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)

tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

🧪 Inference Example
text = "Quote: Be yourself;"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

✅ Expected Output
Quote: Be yourself; everyone else is already taken.
Author: Oscar Wilde

📊 Results

The model was trained for 100 steps

QLoRA enabled training on Google Colab Free Tier (T4 GPU)

Memory usage dramatically reduced using 4-bit NF4 quantization

The model successfully learns to:

Complete partial quotes

Generate full quotes

Attribute the correct author

🤝 Credits

Model: Google DeepMind — Gemma-2B

Dataset: Abirate — english_quotes

Frameworks: Hugging Face Transformers, TRL, PEFT, BitsAndBytes
