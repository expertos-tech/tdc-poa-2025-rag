# Serviço Vetorial — FastAPI (RAG)

API que executa o pipeline RAG da demo: busca vetorial no Qdrant, hidratação no MongoDB e geração de resposta via Azure OpenAI.

## Pré-requisitos
- Python 3.11+ e `python -m venv`.
- MongoDB rodando (ex.: `docker compose -f docker/mongodb/docker-compose.yml up -d`).
- Qdrant rodando (ex.: `docker compose -f docker/qdrant/docker-compose.yml up -d`).
- Variáveis de ambiente em `.env`:
  - `MONGO_URI=mongodb://admin:admin@localhost:27017/?authSource=admin`
  - `QDRANT_URL=http://localhost:6333`
  - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_DEPLOYMENT_NAME` (modelo de chat ativo).

## Como rodar
```bash
cd tdc-vector-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn search_service:app --reload
# Sobe em http://localhost:8000
```

## Endpoints principais
- `POST /ask`: fluxo completo de RAG (Qdrant -> Mongo -> Azure OpenAI) e devolve resposta + fontes + tempos.
- `POST /debug/search`: inspeciona apenas a busca vetorial no Qdrant (sem Mongo/LLM), útil para avaliar relevância.

### Exemplo de chamada `/ask`
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "Quais trilhas de sexta-feira?", "limit": 4}'
```
Resposta inclui `answer` em markdown, `sources` e tempos por etapa em `timings`.

### Exemplo de chamada `/debug/search`
```bash
curl -X POST http://localhost:8000/debug/search \
  -H "Content-Type: application/json" \
  -d '{"text": "Quem fala sobre IA generativa?", "limit": 5}'
```
Retorna hits deduplicados com `mongo_id`, `title`, `type`, `vector_type` e score direto do Qdrant.
