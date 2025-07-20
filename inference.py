import os
from llama_cpp import Llama

def load_model(model_path, context_size=4096, threads=8):
    """
    GGUF modelini yükle
    """
    try:
        print(f"🔄 Model yükleniyor: {os.path.basename(model_path)}")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=context_size,        # context boyutu
            n_threads=threads,         # thread sayısı
            verbose=False,             # verbose çıktı kapalı
            n_gpu_layers=0            # CPU kullanımı için 0, GPU için artırın
        )
        
        print("✅ Model başarıyla yüklendi!")
        return llm
        
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return None

def run_inference(llm, prompt, max_tokens=300, temperature=0.7):
    """
    Inference yap
    """
    try:
        response = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],      # stop token
            echo=False                # prompt'u tekrar gösterme
        )
        
        return response['choices'][0]['text'].strip()
        
    except Exception as e:
        print(f"❌ Inference hatası: {e}")
        return None

def main():
    """
    İnteraktif chat
    """
    print("🚀 GGUF Model İnteraktif Chat")
    print("=" * 40)
    
    # Model yolunu al
    model_path = input("GGUF model dosyası yolu (varsayılan: model.gguf): ").strip()
    if not model_path:
        model_path = "model.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return
    
    # Modeli yükle
    llm = load_model(model_path)
    if not llm:
        return
    
    print("\n💬 İnteraktif Chat Başladı")
    print("Çıkmak için 'q' veya 'quit' yazın")
    print("=" * 40)
    
    while True:
        user_input = input("\n👤 Siz: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'çık', 'exit']:
            print("👋 Görüşmek üzere!")
            break
        
        if not user_input:
            continue
        
        # Prompt formatla
        prompt = f"""<|im_start|>system
Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın.
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""
        
        print("🔄 Düşünüyor...")
        
        response = run_inference(
            llm=llm,
            prompt=prompt,
            max_tokens=300,
            temperature=0.5
        )
        
        if response:
            print(f"🤖 Asistan: {response}")
        else:
            print("❌ Yanıt alınamadı, tekrar deneyin")

if __name__ == "__main__":
    # Önce kütüphaneyi yükle
    try:
        from llama_cpp import Llama
        main()
    except ImportError:
        print("❌ llama-cpp-python kütüphanesi bulunamadı!")
        print("Yüklemek için:")
        print("pip install llama-cpp-python")
        print("\nGPU desteği için:")
        print("pip install llama-cpp-python[cuda]  # NVIDIA GPU")
        print("pip install llama-cpp-python[metal] # Mac M1/M2")

# Kurulum komutları:
"""
# CPU versiyonu:
pip install llama-cpp-python

# CUDA GPU desteği:
pip install llama-cpp-python[cuda]

# Mac Metal desteği:
pip install llama-cpp-python[metal]

# Manuel derleme ile:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
"""