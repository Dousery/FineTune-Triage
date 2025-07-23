# 🇹🇷 Turkish Medical Triage LLaMA3 - GGUF

🔗 **Model Hugging Face Link**: [https://huggingface.co/dousery/turkish-medical-triage-llama3-gguf](https://huggingface.co/dousery/turkish-medical-triage-llama3-gguf)

A realistic Turkish-language **medical triage classification model** built on Meta's LLaMA 3 and quantized to GGUF format for efficient inference.

> ⚕️ This model is fine-tuned to classify emergency medical statements written in Turkish, ideal for triage scenarios where **urgency level prediction** is critical.

---

## 🧠 Model Overview

- **Base Model**: Meta's LLaMA 3
- **Format**: GGUF (compatible with llama.cpp, text-generation-webui, LM Studio, KoboldCpp, etc.)
- **Task**: Turkish medical triage — classify emergency medical complaints into urgency levels.
- **Training Data**: Synthetic yet realistic Turkish emergency complaints, enriched with symptom tags and urgency labels.

### 📌 Example Input/Output
```text
Input: "nefes alamıyorum, göğsümde ağrı var"
Output: ACIL (Urgent)
```

---

## 🚀 Use Cases

- Emergency response systems
- Turkish medical chatbot assistants
- Clinical triage training simulations
- Low-resource language fine-tuning experiments

---

## 🧪 Quantized Model Files

| File | Description | Size |
|------|-------------|------|
| `*.gguf` | Quantized model weights (e.g., Q4_K_M, Q8_0) | Varies |
| `tokenizer.model` | Tokenizer for Turkish text | – |

> GGUF quantized files are optimized for local inference on CPU/GPU with minimal resource usage.

---

## 🔧 Inference Instructions

You can run this model using:

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [LM Studio](https://lmstudio.ai)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

### 🔹 With `llama.cpp`
```bash
./main -m turkish-medical-triage-llama3.Q4_K_M.gguf -p "nefes alamıyorum, başım dönüyor"
```

### 🔹 With `text-generation-webui`

1. Place the `.gguf` file in your `models/` directory.
2. Launch the Web UI and load the model.
3. Use Turkish prompts related to medical symptoms or complaints.

---

## 📊 Dataset Summary

Each data sample includes:

- A Turkish medical complaint sentence
- A list of extracted symptoms
- A target urgency level (`ACIL`, `ORTA`, `NORMAL`, etc.)

#### Example Sample:
```json
{
  "complaint": "çocuğumun ateşi çok yüksek ve kusuyor",
  "symptoms": ["ateş", "kusma"],
  "urgency_level": "ACIL"
}
```

---

## 💡 Motivation

This model was created to improve triage automation in Turkish healthcare scenarios and to provide open-source resources in low-resource medical NLP fields.

---

## 📁 Files & Structure

- `README.md` – Model description and usage
- `*.gguf` – Quantized model files
- `tokenizer.model` – Tokenizer used for Turkish language inputs

---

## ✍️ Author

- [Doğuser Yarar](https://huggingface.co/dousery)  
🧪 AI / RPA enthusiast | Turkish NLP & Healthcare AI researcher

---

## 📜 License

[MIT License](LICENSE)  
**Disclaimer**: This model is for research purposes only and **not intended for real-life diagnosis or emergency decision-making**.

---

> 💬 For any feedback, issues, or contributions, feel free to open an issue or reach out!