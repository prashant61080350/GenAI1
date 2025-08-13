# No-Memory Chatbot (Full-Stack)

A minimal full-stack chatbot with:
- Frontend: React + Vite
- Backend: FastAPI
- LLM: pluggable backends (mock, Hugging Face Inference API, or local Transformers). No memory — each message is independent.

## Run backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

To use a real model, edit `.env`:
- `MODEL_BACKEND=ollama` and set `OLLAMA_MODEL` (ensure Ollama is running and the model is pulled)
- or `MODEL_BACKEND=hf_api` and set `HF_API_MODEL`, `HF_API_TOKEN`
- or `MODEL_BACKEND=local` and set `MODEL_NAME` (install `transformers` and `torch`)

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend dev server proxies `/api` to `http://localhost:8000`.

## Build frontend
```bash
cd frontend
npm run build && npm run preview
```

## Notes
- No conversation memory is stored; each request is independent.
- Optional SSE streaming is implemented at `GET /api/stream`.
- Basic error handling and logging included.