# TDC Experience Porto Alegre 2025 — RAG de ponta a ponta

Repositório da apresentação “Sua IA, seus dados: construindo um RAG de verdade”. Aqui ficam os artefatos usados na demo que mostra como colocar dados próprios dentro de uma IA: preparação, chunking, vetorização, armazenamento vetorial e integração com LLM.

## O que há em cada pasta
- `tdc-ai-assistant/`: frontend Angular que chama o serviço de busca e exibe respostas com traço de origem.
- `tdc-data-loader/`: script Node.js que carrega as palestras e infos do evento no MongoDB.
- `tdc-vector-service/`: API FastAPI que executa o fluxo RAG (Qdrant + Mongo + Azure OpenAI) e expõe endpoints `/ask` e `/debug/search`.
- `docker/`: compose files para subir MongoDB + Mongo Express e Qdrant localmente.

## Como rodar o stack completo (visão rápida)
1) Suba as dependências: `docker compose -f docker/mongodb/docker-compose.yml up -d` e `docker compose -f docker/qdrant/docker-compose.yml up -d`.
2) Carregue os dados: `cd tdc-data-loader && npm install && node seed.js` (usa `mongodb://admin:admin@localhost:27017/?authSource=admin`).
3) Inicie o serviço vetorial: `cd tdc-vector-service && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && uvicorn search_service:app --reload`.
4) Rode o frontend: `cd tdc-ai-assistant && npm install && npm start` e acesse `http://localhost:4200`.
