# Repository Guidelines

## Project Structure & Module Organization
- `tdc-ai-assistant/`: Angular 19 frontend; source in `src/`, assets in `public/`, builds into `dist/`.
- `tdc-data-loader/`: Node.js seed script (`seed.js`) that loads `data/*.js` into MongoDB.
- `tdc-vector-service/`: FastAPI service (`search_service.py`) that queries Qdrant + Mongo and calls Azure OpenAI; debug helpers live here too.
- `docker/`: Compose stacks for infra (`docker/mongodb`, `docker/qdrant`) to start local MongoDB, Mongo Express, and Qdrant.

## Build, Test, and Development Commands
- Frontend: `cd tdc-ai-assistant && npm install && npm start` (dev server on `http://localhost:4200`), `npm run build` (production bundle), `npm test` (Karma).
- Data loader: `cd tdc-data-loader && npm install && node seed.js` (expects MongoDB at `mongodb://admin:admin@localhost:27017/?authSource=admin`).
- Vector service: `cd tdc-vector-service && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && uvicorn search_service:app --reload`.
- Infra: `docker compose -f docker/mongodb/docker-compose.yml up -d` and `docker compose -f docker/qdrant/docker-compose.yml up -d` to start dependencies.

## Coding Style & Naming Conventions
- TypeScript/HTML/SCSS: Angular CLI defaults; 2-space indentation, strict typing, `FeatureNameComponent` in `*.component.ts` with `*.spec.ts` siblings.
- Python: PEP8 with snake_case names; keep env access via `os.getenv` and avoid hard-coded secrets.
- JavaScript (data loader): CommonJS modules; prefer async/await and concise logging aligned with existing Portuguese commentary.
- Config lives in `.env` (`MONGO_URI`, `QDRANT_URL`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_DEPLOYMENT_NAME`); never commit secrets.

## Testing Guidelines
- Frontend: keep specs beside components (`*.spec.ts`), run `npm test`; add integration cases for complex flows.
- Backend: add pytest tests under `tdc-vector-service/tests/` named `test_*.py`; mock Mongo/Qdrant/Azure to stay hermetic.
- Data loader lacks automated tests; after `node seed.js`, verify collection counts.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (e.g., `Add qdrant bootstrap script`); reference issues when available.
- PRs should note scope, commands run, and infra needed (`docker compose`, env vars); include screenshots/GIFs for UI changes.
- Ensure `npm test` and new backend tests pass; state any manual checks performed.
- Keep diffs focused: separate feature, infra, and formatting changes when practical.

## Security & Configuration Tips
- Default connection strings use local credentials; override via `.env` for other environments.
- Avoid logging secrets or full payloads from Azure OpenAI; keep timing/status logs already present in `search_service.py`.
