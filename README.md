# 🏥 Turkish Medical Emergency Triage Model

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/dousery/llama3-turkish-medical-triage)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Turkish](https://img.shields.io/badge/Language-Turkish-red)](https://github.com/topics/turkish)
[![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI-green)](https://github.com/topics/medical-ai)

> **Llama-3 8B** tabanlı Türkçe tıbbi aciliyet değerlendirme modeli. LoRA fine-tuning ile hasta şikayetlerini analiz eder ve aciliyet seviyesi belirler.

## 🚀 Hızlı Başlangıç

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ve tokenizer yükle
model_name = "dousery/llama3-turkish-medical-triage"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Örnek kullanım
prompt = """<|im_start|>system
Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın.
<|im_end|>
<|im_start|>user
Hasta şikayeti: Göğsümde şiddetli ağrı var, nefes almakta zorlanıyorum
<|im_end|>
<|im_start|>assistant"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
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
