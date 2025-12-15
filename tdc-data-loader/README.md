# Data Loader — MongoDB Seed

Script usado na demo para criar o “Data Lake” do RAG com palestras e informações do evento.

## Pré-requisitos
- MongoDB local rodando com usuário/senha `admin/admin` (use `docker compose -f docker/mongodb/docker-compose.yml up -d` a partir da raiz).
- Node.js 18+.

## Como executar
```bash
cd tdc-data-loader
npm install
node seed.js
```
O script conecta em `mongodb://admin:admin@localhost:27017/?authSource=admin`, limpa as coleções e insere:
- `talks` com as palestras (dia 1 e 2).
- `event_info` com metadados do evento.

## Onde ficam os dados-fonte
- `data/talks_day1.js` e `data/talks_day2.js`: conteúdo das sessões.
- `data/event_info.js`: dados gerais do TDC Experience Porto Alegre 2025.
