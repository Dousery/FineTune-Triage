import modal

app = modal.App("lora-merge-medical")

volume = modal.Volume.from_name("medical-finetune-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "transformers", "accelerate", "unsloth", "bitsandbytes"
    )
)

@app.function(image=image, volumes={"/root/vol": volume}, gpu="A100", timeout=3000)
def merge_and_save_model():
    from unsloth import FastLanguageModel
    import torch
    import os

    MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    ADAPTER_PATH = "/root/vol/finetuned"
    SAVE_PATH = "/root/vol/merged_model"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

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

    model.load_adapter(ADAPTER_PATH, adapter_name="default")

    merged_model = model.merge_and_unload()

    print(f"Attempting to save model to {SAVE_PATH}")
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        merged_model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        
        print(f"✅ Model files saved successfully to {SAVE_PATH}")
        print("Contents of save directory:")
        for item in os.listdir(SAVE_PATH):
            file_path = os.path.join(SAVE_PATH, item)
            size = os.path.getsize(file_path)
            print(f"- {item} ({size} bytes)")
            
        essential_files = ["config.json", "model.safetensors.index.json", "tokenizer.json"]
        missing_files = [f for f in essential_files if not os.path.exists(os.path.join(SAVE_PATH, f))]
        if missing_files:
            raise Exception(f"Missing essential files: {', '.join(missing_files)}")
            
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        raise

    print("✅ Merge işlemi tamamlandı. Model şurada kaydedildi:", SAVE_PATH)
    return SAVE_PATH


@app.function(volumes={"/root/vol": volume})
def create_download_archive():
    import tarfile
    import io
    import os
    
    MODEL_PATH = "/root/vol/merged_model"
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Merged model not found at {MODEL_PATH}")
    
    print("Creating download archive...")
    
    tar_buffer = io.BytesIO()
    
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        tar.add(MODEL_PATH, arcname='merged_model')
    
    tar_buffer.seek(0)
    archive_data = tar_buffer.getvalue()
    
    print(f"✅ Archive created. Size: {len(archive_data)} bytes")
    return archive_data


@app.function(volumes={"/root/vol": volume})
def get_model_info():
    import os
    import json
    
    MODEL_PATH = "/root/vol/merged_model"
    
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found"}
    
    info = {
        "model_path": MODEL_PATH,
        "files": [],
        "total_size": 0
    }
    
    for root, dirs, files in os.walk(MODEL_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, MODEL_PATH)
            size = os.path.getsize(file_path)
            
            info["files"].append({
                "name": relative_path,
                "size": size
            })
            info["total_size"] += size
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            info["config"] = json.load(f)
    
    return info


@app.local_entrypoint()
def main():
    import os

    print("1. Model merge işlemi başlatılıyor...")
    model_path = merge_and_save_model.remote()
    
    print("\n2. Model bilgileri alınıyor...")
    model_info = get_model_info.remote()
    
    if "error" in model_info:
        print(f"❌ {model_info['error']}")
        return
    
    print(f"📊 Model Bilgileri:")
    print(f"   - Toplam dosya sayısı: {len(model_info['files'])}")
    print(f"   - Toplam boyut: {model_info['total_size'] / (1024**3):.2f} GB")
    print(f"   - Ana dosyalar:")
    for file in model_info['files'][:5]:
        print(f"     * {file['name']}: {file['size'] / (1024**2):.1f} MB")
    
    print("\n📥 Model indiriliyor...")
    archive_data = create_download_archive.remote()
    
    local_filename = "merged_medical_model.tar.gz"
    with open(local_filename, 'wb') as f:
        f.write(archive_data)
    
    print(f"✅ Model indirildi: {local_filename}")
    print(f"   Boyut: {len(archive_data) / (1024**2):.1f} MB")
    
    import tarfile
    with tarfile.open(local_filename, 'r:gz') as tar:
        tar.extractall('.')
    
    print("✅ Model dosyaları çıkarıldı: ./merged_model/")