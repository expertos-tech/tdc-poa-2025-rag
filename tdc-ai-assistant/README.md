# Frontend — TDC AI Assistant (Angular 19)

Interface da demo “Sua IA, seus dados: construindo um RAG de verdade” que consome o serviço vetorial e mostra respostas com traço de origem.

## Como rodar em desenvolvimento
```bash
cd tdc-ai-assistant
npm install
npm start          # ou: ng serve
# Acesse http://localhost:4200
```
O app espera o backend em `http://localhost:8000` (ajuste no código se necessário).

## Build de produção
```bash
npm run build
```
Artefatos ficam em `dist/tdc-ai-assistent/`.

## Testes
```bash
npm test
```
Executa os specs Karma/Jasmine.
