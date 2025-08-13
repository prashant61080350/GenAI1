# No-Memory Chatbot Backend (FastAPI)

## Quick start

```bash
cd backend
pip install -r requirements.txt --break-system-packages
cp .env.example .env
# Ensure Ollama is running and a model is pulled (e.g., llama3.2:1b):
# ollama pull llama3.2:1b
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /api/health` – health check
- `POST /api/chat` – JSON: `{ "message": "..." }` -> `{ "reply": "..." }`
- `GET /api/stream?message=...` – Server-Sent Events token stream

## Configure an LLM backend

This project is configured to run without heavy dependencies by default.
Set `MODEL_BACKEND` in `.env` to control behavior:

- `ollama` (default): uses a local open-source LLM served by Ollama (`OLLAMA_MODEL`, `OLLAMA_HOST`)
- `hf_api`: uses Hugging Face Inference API (set `HF_API_MODEL`, `HF_API_TOKEN`)
- `local`: uses Hugging Face Transformers locally (set `MODEL_NAME`)
- `mock`: returns a placeholder echo for quick testing

### HF Inference API backend
1. Ensure `huggingface_hub` is installed (already in requirements)
2. Set in `.env`:
```
MODEL_BACKEND=hf_api
HF_API_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_API_TOKEN=hf_xxx
```

### Local Transformers backend
1. Install heavy deps (optional):
```
pip install transformers torch
```
2. Set in `.env`:
```
MODEL_BACKEND=local
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Note: large models may require a GPU. For CPU-only testing, you can try smaller models like `sshleifer/tiny-gpt2` for demonstration purposes.