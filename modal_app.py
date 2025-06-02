import modal
import os
import shutil
from pathlib import Path

# ----------- Modal Setup -----------
app = modal.App("medical-finetune")
volume = modal.Volume.from_name("medical-finetune-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "transformers", "accelerate", "unsloth", "datasets", "bitsandbytes"
    )
)

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 2048

@app.function(gpu="A100", image=image, volumes={"/root/vol": volume})
def train():
    import torch, json, gc
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer
    from unsloth import FastLanguageModel

    def clear_memory():
        gc.collect()
        torch.cuda.empty_cache()

    clear_memory()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    with open("/root/vol/medical_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def create_prompt(item):
        symptoms = ", ".join(item.get('symptoms', []))
        return f"""<|im_start|>system\nSen tıbbi aciliyet değerlendirmesi yapan bir asistansın.\n<|im_end|>\n<|im_start|>user\nHasta şikayeti: {item.get('input_text', '')}\nTespit edilen semptomlar: {symptoms}\n<|im_end|>\n<|im_start|>assistant\nAciliyet Seviyesi: {item.get('urgency_label', '')}\nÖneriler: {item.get('response', '')}\nDeğerlendirme: {item.get('reasoning', '')}\n<|im_end|>"""

    dataset = Dataset.from_dict({"text": [create_prompt(x) for x in raw_data]})

    def tokenize_function(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],
        batched=False
    )

    training_args = TrainingArguments(
        output_dir="/root/vol/finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=30,
        logging_steps=5,
        save_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model("/root/vol/finetuned")
    tokenizer.save_pretrained("/root/vol/finetuned")

@app.function(gpu="A100", image=image, volumes={"/root/vol": volume})
def infer(prompt: str) -> str:
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    import torch

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Apply LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load the fine-tuned weights
    model.load_adapter("/root/vol/finetuned", adapter_name="default")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

@app.local_entrypoint()
def main():
    # Create vol directory if it doesn't exist
    vol_dir = Path("vol")
    vol_dir.mkdir(exist_ok=True)
    
    # Copy medical data to vol directory
    source_file = Path("medical_data.json")
    target_file = vol_dir / "medical_data.json"
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file {source_file} does not exist!")
        
    shutil.copy2(source_file, target_file)
    print(f"Copied {source_file} to {target_file}")
    
    # Start training
    print("Starting training...")
    train.remote()
    
    # Run inference test
    print("Running inference test...")
    prompt = """<|im_start|>system\nSen tıbbi aciliyet değerlendirmesi yapan bir asistansın.\n<|im_end|>\n<|im_start|>user\nHasta şikayeti: Nefes darlığı ve göğüs ağrısı var.\n<|im_end|>\n<|im_start|>assistant\n"""
    response = infer.remote(prompt)
    print("Model Yanıtı:\n", response)