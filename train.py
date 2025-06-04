import modal
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
import json

# Create a Modal app
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

def format_prompt(example):
    prompt = f"""<|im_start|>system
Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın.
<|im_end|>
<|im_start|>user
{example['input_text']}
<|im_end|>
<|im_start|>assistant
{example['response']}
<|im_end|>"""
    return {"text": prompt}

@app.function(gpu="A100", image=image, volumes={"/root/vol": volume})
def train():
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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and prepare dataset
    with open("medical_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset = load_dataset("json", data_files={"train": "medical_data.json"})
    
    # Format prompts
    dataset = dataset.map(format_prompt)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/root/vol/finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
    )

    # Train the model
    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        args=training_args,
    )

    trainer.train()

    # Save the model
    trainer.save_model("/root/vol/finetuned")

@app.local_entrypoint()
def main():
    train.remote() 