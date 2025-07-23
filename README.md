# 🇹🇷 Turkish Medical Triage LLaMA3 - GGUF

🔗 **Model Hugging Face Link**: [https://huggingface.co/dousery/turkish-medical-triage-llama3-gguf](https://huggingface.co/dousery/turkish-medical-triage-llama3-gguf)

<img width="1611" height="670" alt="image" src="https://github.com/user-attachments/assets/04772f33-ec0c-44db-9a64-4b0a5a16a847" />


An open-source Turkish-language **medical triage and diagnosis model**, fine-tuned on synthetic emergency medical data. Built on Meta's LLaMA 3 and quantized to GGUF format for efficient offline inference.

> ⚕️ This model predicts both the **urgency level** of a Turkish medical complaint and provides a **clear response message**, emulating how a virtual assistant might reply in a triage scenario.

---

## 🧠 Model Capabilities

- 🔍 **Input**: Free-form Turkish medical complaint sentence
- 🎯 **Output**: Natural-language response that includes:
  - Triage level (e.g. ACIL, ORTA, NORMAL)
  - Possible diagnosis
  - Clear, directive language in Turkish

### 📌 Example

```json
{
  "input": "nefes alamıyorum, göğsümde batıcı ağrı var ve terliyorum",
  "response": "ACIL DURUM: Anında 112'yi arayın. Kalp krizi belirtileri gösteriyorsunuz. Derhal hastaneye gitmeniz gerekiyor."
}
```

---

## 🚀 Use Cases

- Emergency virtual triage agents
- Turkish healthcare chatbots
- Simulation for training emergency responders
- Research in multilingual healthcare LLMs

---

## 🔧 Inference with GGUF

You can run this model using:

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [LM Studio](https://lmstudio.ai)

### 🖥️ Example with `llama.cpp`:

```bash
./main -m turkish-medical-triage-llama3.Q4_K_M.gguf -p "başım çok dönüyor, ayakta duramıyorum ve kusuyorum"
```

Expected Output:
```
ORTA DERECE RİSK: Muhtemelen bir iç kulak enfeksiyonu veya mide rahatsızlığı yaşıyor olabilirsiniz. En kısa sürede bir sağlık kuruluşuna başvurmanız önerilir.
```

---

## 📊 Dataset Summary

Each training sample includes:

- A Turkish-language medical complaint sentence
- Annotated symptoms
- A possible diagnosis
- A natural-language response combining triage + advice

---

## 📁 Files

- `README.md`
- `*.gguf` – Quantized model files
- `tokenizer.model`

---

## ✍️ Author

- [Doğuser Yarar](https://huggingface.co/dousery)  
🧪 Turkish NLP / AI researcher focused on public-benefit applications

---

## 📜 License

[MIT License](LICENSE)

⚠️ **Disclaimer**: This model is not intended for real-life clinical use. Do not use for medical decisions without professional supervision.

---

> 💬 Suggestions, feedback, and contributions are welcome!
