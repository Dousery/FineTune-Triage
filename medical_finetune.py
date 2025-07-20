# medical_finetune.py
import modal
import os
import json

app = modal.App("turkish-medical-triage-finetune")

# Use the corrected Dockerfile path or build from string
unsloth_image = modal.Image.from_dockerfile("./DockerFile")  # Make sure Dockerfile is in the same directory

# Add the data file to the image
unsloth_image = unsloth_image.add_local_file("vol/medical_data.json", "/root/data/medical_data.json")

volume = modal.Volume.from_name("turkish-medical-triage-finetune-vol", create_if_missing=True)

@app.function(
    gpu="A100",
    image=unsloth_image,
    timeout=60 * 60 * 6,
    volumes={"/root/vol": volume},
    secrets=[modal.Secret.from_name("hugginface-secret")],
)
def train():
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch

    base_model = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length = 2048
    hf_save_gguf_name = "dousery/turkish-medical-triage-llama3-gguf"  # Huggingface reposu
    quantization_method = "q4_k_m"

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format prompt
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

    # Load dataset
    dataset = load_dataset("json", data_files={"train": "/root/data/medical_data.json"})  # Updated path
    dataset = dataset.map(format_prompt)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # Trainer config
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_seq_length=max_seq_length,
            num_train_epochs=3,
            warmup_steps=5,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="/root/vol/finetuned",
            report_to="none",
            dataset_text_field="text",
        )
    )

    # Train & Save
    trainer.train()

    # Push to HuggingFace
    model.push_to_hub_gguf(
        hf_save_gguf_name,
        tokenizer,
        quantization_method=quantization_method,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

@app.local_entrypoint()
def main():
    train.remote()
    print("Training job submitted successfully!")