Fine-Tuning Gemma-2b for Quote Generation 🧠✨This repository demonstrates how to fine-tune Google's Gemma-2b Large Language Model (LLM) using QLoRA (Quantized Low-Rank Adaptation) on a consumer GPU (e.g., Google Colab T4).The model is trained on the Abirate/english_quotes dataset to memorize quotes and correctly attribute them to their authors.🚀 Project OverviewBase Model: google/gemma-2bTechnique: QLoRA (4-bit quantization + LoRA adapters)Library: Hugging Face trl (SFTTrainer), peft, bitsandbytesObjective: Input a partial quote $\rightarrow$ Model completes the quote and provides the author name.🛠️ Installation & RequirementsThis project relies on specific versions of Hugging Face libraries to ensure compatibility with TRL.Bashpip install -q --upgrade torch
pip install -q --upgrade transformers==4.38.2
pip install -q --upgrade trl==0.7.10
pip install -q --upgrade peft==0.8.2
pip install -q --upgrade bitsandbytes==0.42.0
pip install -q --upgrade accelerate==0.27.2
📂 DatasetThe model is trained on the English Quotes dataset. The data is pre-processed to follow this specific format:PlaintextQuote: {quote_text}
Author: {author_name}
⚙️ Training ConfigurationThe training uses Parameter Efficient Fine-Tuning (PEFT) to update only a fraction of the model's parameters, making training fast and memory-efficient.LoRA Rank (r): 8Target Modules: q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_projQuantization: 4-bit (NF4)Optimizer: paged_adamw_8bit💻 Usage Code1. Loading the ModelPythonimport torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
2. Inference (Testing)After training, you can generate text like this:Pythontext = "Quote: Be yourself;"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Expected Output:Quote: Be yourself; everyone else is already taken.Author: Oscar Wilde📊 ResultsThe model was fine-tuned for 100 steps. By using 4-bit quantization, the memory footprint is significantly reduced, allowing this training to run on the free tier of Google Colab.
🤝 CreditsModel by Google DeepMind
Dataset by Abirate
