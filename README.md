# 🏥 Turkish Medical Emergency Triage Model

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/dousery/turkish-medical-triage-llama3-gguf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Turkish](https://img.shields.io/badge/Language-Turkish-red)](https://github.com/topics/turkish)
[![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-green)](https://github.com/topics/medical-ai)

> **Llama-3 8B** tabanlı Türkçe tıbbi aciliyet değerlendirme modeli. LoRA fine-tuning ile hasta şikayetlerini analiz eder ve aciliyet seviyesi belirler.

## 🚀 Kullanım

```python
import os
from llama_cpp import Llama

def load_model(path):
    try:
        print(f"🔄 Model yükleniyor: {os.path.basename(path)}")
        model = Llama(model_path=path, n_ctx=4096, n_threads=8, verbose=False, n_gpu_layers=0)
        print("✅ Model yüklendi")
        return model
    except Exception as e:
        print(f"❌ Yükleme hatası: {e}")
        return None

def run_inference(model, prompt):
    try:
        result = model(prompt=prompt, max_tokens=300, temperature=0.5, stop=["<|im_end|>"], echo=False)
        return result['choices'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Inference hatası: {e}")
        return None

def main():
    print("🚀 GGUF Model Chat - Çıkmak için 'q' yaz")
    path = input("Model dosya yolu (varsayılan: model.gguf): ").strip() or "model.gguf"

    if not os.path.exists(path):
        print(f"❌ Dosya bulunamadı: {path}")
        return

    model = load_model(path)
    if not model:
        return

    while True:
        user_input = input("\n👤 Siz: ").strip()
        if user_input.lower() in ['q', 'quit', 'çık', 'exit']:
            break
        if not user_input:
            continue

        prompt = f"""<|im_start|>system
Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın.
<|im_end|>
<|im_start|>user
{user_input}
<|im_end|>
<|im_start|>assistant
"""

        print("🔄 Düşünüyor...")
        response = run_inference(model, prompt)
        print(f"🤖 Asistan: {response}" if response else "❌ Yanıt alınamadı")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("❌ llama-cpp-python eksik! Yüklemek için:\n")
        print("pip install llama-cpp-python")

```

## ⚡ Özellikler

- 🇹🇷 **Türkçe Optimize**: Native Türkçe tıbbi terminoloji
- 🎯 **Aciliyet Sınıflandırması**: Çok Acil - Acil - Normal arası değerlendirme
- 🧠 **LoRA Fine-tuned**: Efficient training ile optimize edildi
- 📋 **Triage Sistemi**: Hasta önceliklendirmesi için tasarlandı
- ⚡ **Hızlı İnference**: 8B parametreli efficient model

## 🏗️ Model Detayları

| Özellik | Değer |
|---------|-------|
| **Base Model** | `unsloth/llama-3-8b-bnb-4bit` |
| **Fine-tuning** | LoRA (r=16, α=16) |
| **Dil** | Türkçe |
| **Max Length** | 2048 tokens |
| **Boyut** | ~15GB |

## 🎭 Örnek Çıktılar

**Input**: "Başım çok ağrıyor, kusma var, ışık gözümü yakıyor"

**Output**:
```
Aciliyet Seviyesi: Yüksek
Öneriler: Acil servise başvurun, nörolojik muayene gerekli
Değerlendirme: Migren veya intrakranial basınç artışı olasılığı
```

## ⚠️ Önemli Uyarılar

> **🚨 SADECE EĞİTİM VE ARAŞTIRMA AMAÇLIDIR**
> 
> - Gerçek tıbbi durumlar için kullanmayın
> - Profesyonel tıbbi tavsiye yerine geçmez
> - Acil durumlarda **112**'yi arayın

## 📊 Kullanım Alanları

✅ **Uygun**: Tıbbi eğitim, araştırma, simülasyon  
❌ **Uygun Değil**: Gerçek hasta değerlendirmesi, teşhis

## 📄 Lisans

Apache 2.0 License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🔗 Bağlantılar

- 🤗 [Hugging Face Model](https://huggingface.co/dousery/llama3-turkish-medical-triage)
- 📧 İletişim: [GitHub Issues](https://github.com/dousery/FineTune-Triage/issues)

---

<div align="center">

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

*Türkiye'de tıbbi AI araştırmalara katkıda bulunması için oluşturulmuştur.*

</div>
