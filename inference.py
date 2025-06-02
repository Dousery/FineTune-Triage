import modal

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

    # Load the fine-tuned weights
    model.load_adapter("/root/vol/finetuned", adapter_name="default")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get the <|im_end|> token ID to stop generation correctly
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_token_id,
    )

    # Decode and clean up the output
    decoded = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    cleaned = decoded.split("<|im_end|>")[0].strip()

    return cleaned


@app.local_entrypoint()
def main():
    prompt = """<|im_start|>system
Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın.
<|im_end|>
<|im_start|>user
Hasta şikayeti: Yemek yedikten sonra karnımda şiddetli bir ağrı var, şişkinlik ve gaz da eşlik ediyor
<|im_end|>
<|im_start|>assistant
"""
    response = infer.remote(prompt)
    print("Model Yanıtı:", response)
