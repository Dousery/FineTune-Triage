#!/usr/bin/env python3
"""
Tıbbi Aciliyet Değerlendirme Modeli - Hugging Face Upload
Modal'da eğitilmiş merge edilmiş modeli HF Hub'a yükler
"""

import os
import json
from datetime import datetime
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_medical_model_card(model_info, repo_name):
    """Tıbbi model için özel model card oluştur"""
    
    readme_content = f"""---
language:
- tr
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- medical
- turkish
- emergency
- triage
- fine-tuned
- lora
- healthcare
base_model: unsloth/llama-3-8b-bnb-4bit
datasets:
- medical-emergency-triage
model-index:
- name: {repo_name.split('/')[-1]}
  results: []
---

# 🏥 Tıbbi Aciliyet Değerlendirme Modeli

Bu model, **Llama-3-8B** temel modeli üzerine **LoRA (Low-Rank Adaptation)** yöntemi ile Türkçe tıbbi aciliyet verileri üzerinde fine-tune edilmiştir. Model, hasta şikayetlerini analiz ederek aciliyet seviyesi değerlendirmesi yapar.

## 🎯 Model Özellikleri

- **Temel Model**: `unsloth/llama-3-8b-bnb-4bit`
- **Fine-tuning Yöntemi**: LoRA (r=16, alpha=16)
- **Dil**: Türkçe ve İngilizce
- **Domain**: Tıbbi Aciliyet Değerlendirmesi
- **Boyut**: ~{model_info.get('total_size_gb', 15):.1f} GB
- **Maksimum Sequence Length**: 2048 tokens

## 🚨 Önemli Uyarı

⚠️ **Bu model sadece eğitim ve araştırma amaçlıdır!**

- Gerçek tıbbi durumlar için kullanılmamalıdır
- Profesyonel tıbbi tavsiye yerine geçmez  
- Acil durumlarda 112'yi arayın
- Model çıktıları her zaman doğrulanmalıdır

## 📋 Kullanım Alanları

✅ **Uygun Kullanım:**
- Tıbbi eğitim simülasyonları
- Araştırma projeleri
- Triage algoritmaları geliştirme
- Tıbbi NLP araştırmaları

❌ **Uygun Olmayan Kullanım:**
- Gerçek hasta değerlendirmesi
- Klinik karar verme
- Teşhis koyma
- Tedavi önerme


## 🔧 Teknik Detaylar

### LoRA Konfigürasyonu
- **Rank (r)**: 16
- **Alpha**: 16
- **Dropout**: 0.1
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Bias**: none

### Eğitim Parametreleri
- **Batch Size**: 1 (gradient accumulation: 4)
- **Learning Rate**: 2e-4
- **Max Steps**: 30
- **Optimizer**: AdamW
- **Precision**: FP16/BF16
- **Max Length**: 512 tokens

### Prompt Formatı

Model aşağıdaki format ile eğitilmiştir:

```
<|im_start|>system
Sen tıbbi aciliyet değerlendirmesi yapan bir asistansın.
<|im_end|>
<|im_start|>user
Hasta şikayeti: [ŞİKAYET]
Tespit edilen semptomlar: [SEMPTOMLAR]
<|im_end|>
<|im_start|>assistant
Aciliyet Seviyesi: [SEVİYE]
Öneriler: [ÖNERİLER]
Değerlendirme: [AÇIKLAMA]
<|im_end|>
```

## 📊 Model Performansı

### Aciliyet Seviyeleri
- **Çok acil**: Acil müdahale gerekli
- **Acil**: Hızlı değerlendirme gerekli
- **Normal**: Rutin muayene yeterli

### Örnek Değerlendirmeler

**Göğüs Ağrısı:**
```
Aciliyet Seviyesi: Yüksek
Öneriler: Acil servise başvurun, EKG çekilmeli
Değerlendirme: Kardiyak olay riski yüksek
```

**Baş Ağrısı:**
```
Aciliyet Seviyesi: Orta
Öneriler: Nöroloji konsültasyonu önerilir
Değerlendirme: Şiddetli baş ağrısı ciddi sebepleri olabilir
```

## 🛡️ Güvenlik ve Sorumluluk

### Sınırlamalar
- Model çıktıları %100 doğru değildir
- Nadir hastalıkları tanımada yetersiz olabilir
- Kültürel ve bölgesel farklılıkları tam yansıtmayabilir
- Sürekli güncelleme ve iyileştirme gerektirir

### Etik Kullanım
- Hasta mahremiyetini koruyun
- Model önyargılarını göz önünde bulundurun
- Şeffaflık ve açıklanabilirlik sağlayın
- Düzenli performans değerlendirmesi yapın

## 📚 Akademik Kullanım

### Citation

```bibtex
@misc{{medical_emergency_llama3,
    title={{Tıbbi Aciliyet Değerlendirme Modeli - Llama-3 Turkish Medical}},
    author={{[Doguser Yarar]}},
    year={{2025}},
    publisher={{Hugging Face}},
    howpublished={{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```

## 🔄 Model Güncellemeleri

**v1.0** (Aralık 2024)
- İlk release
- Temel aciliyet değerlendirmesi
- Türkçe tıbbi terminoloji desteği


## 📄 Lisans

Apache 2.0 - Açık kaynak kullanım için uygun

---

**🏥 Sağlık alanında AI kullanımı hassas bir konudur. Bu modeli kullanırken sorumlu olun!**

**📅 Son Güncelleme**: {datetime.now().strftime('%d %B %Y')}  
**🔧 Framework**: Transformers, Unsloth, LoRA  
**⚡ Training**: Modal GPU A100
"""

    return readme_content

def upload_medical_model(model_path, username, model_name="llama3-medical-turkish-emergency"):
    """Tıbbi modeli HF Hub'a yükle"""
    
    print("🏥 Tıbbi Aciliyet Modeli Upload Süreci")
    print("=" * 50)
    
    # Repository bilgileri
    repo_name = f"{username}/{model_name}"
    
    print(f"📋 Upload Bilgileri:")
    print(f"   - Model: {model_path}")
    print(f"   - Repository: {repo_name}")
    print(f"   - Visibility: Public")
    print(f"   - Domain: Medical Emergency Triage")
    
    # Model info topla
    model_info = {}
    if os.path.exists(model_path):
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                        for dirpath, dirnames, filenames in os.walk(model_path) 
                        for filename in filenames)
        model_info['total_size_gb'] = total_size / (1024**3)
    
    try:
        # HF API
        api = HfApi()
        
        # Repository oluştur (public)
        print("📁 Public repository oluşturuluyor...")
        api.create_repo(
            repo_id=repo_name,
            private=False,  # Public yap
            repo_type="model"
        )
        print("✅ Repository oluşturuldu")
        
        # Model dosyalarını yükle
        print("📤 Model dosyaları yükleniyor...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message="🏥 Upload Turkish Medical Emergency Triage Model"
        )
        print("✅ Model dosyaları yüklendi")
        
        # README oluştur ve yükle
        print("📝 Detaylı model card oluşturuluyor...")
        readme_content = create_medical_model_card(model_info, repo_name)
        
        with open("README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
            commit_message="📋 Add comprehensive medical model documentation"
        )
        
        # Geçici dosyayı sil
        os.remove("README.md")
        
        print("✅ Model card yüklendi")
        
        # Tags ekle
        print("🏷️ Tags güncelleniyor...")
        api.update_repo_settings(
            repo_id=repo_name,
            repo_type="model",
            private=False
        )
        
        # Tags'i ayrı bir commit olarak ekle
        api.upload_file(
            path_or_fileobj=json.dumps({"tags": ["medical", "turkish", "emergency", "triage", "llama", "lora", "healthcare"]}),
            path_in_repo="tags.json",
            repo_id=repo_name,
            repo_type="model",
            commit_message="🏷️ Add model tags"
        )
        
        print("🎉 Upload başarılı!")
        print(f"🔗 Model linki: https://huggingface.co/{repo_name}")
        print(f"🧪 Test için: AutoModelForCausalLM.from_pretrained('{repo_name}')")
        
        return True
        
    except Exception as e:
        if "already exists" in str(e):
            print("⚠️  Repository zaten mevcut, güncelleniyor...")
            try:
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_name,
                    commit_message="🔄 Update medical model files"
                )
                print("✅ Model güncellendi!")
                return True
            except Exception as update_error:
                print(f"❌ Güncelleme hatası: {update_error}")
                return False
        else:
            print(f"❌ Upload hatası: {e}")
            return False

def main():
    """Ana upload süreci"""
    
    print("🤗 Hugging Face Login")
    print("Token almak için: https://huggingface.co/settings/tokens")
    token = input("HF Token (write yetkili): ").strip()
    
    # Login
    login(token=token)
    print("✅ Giriş başarılı!")
    
    # Model yolu
    model_path = input("\\nModel klasörü yolu (örn: ./merged_model): ").strip()
    if not os.path.exists(model_path):
        print(f"❌ Model klasörü bulunamadı: {model_path}")
        return
    
    # Kullanıcı bilgileri
    username = input("Hugging Face kullanıcı adınız: ").strip()
    
    # Özel model adı öner
    default_name = "llama3-medical-turkish-emergency"
    custom_name = input(f"Model adı (default: {default_name}): ").strip()
    model_name = custom_name if custom_name else default_name
    
    # Onay
    repo_name = f"{username}/{model_name}"
    confirm = input(f"\\n{repo_name} adıyla PUBLIC upload yapmak istediğinizi onaylıyor musunuz? (y/N): ")
    
    if not confirm.lower().startswith('y'):
        print("❌ Upload iptal edildi")
        return
    
    # Upload
    success = upload_medical_model(model_path, username, model_name)
    
    if success:
        print("\\n🎊 Tıbbi modeliniz başarıyla yüklendi!")
        print(f"🌍 Public erişim: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    main()