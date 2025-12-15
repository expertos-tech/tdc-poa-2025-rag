import os
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- SETUP INICIAL ---
load_dotenv()
app = FastAPI(title="TDC Vector Search Service")

# 1. Configuração de Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 API Iniciada. Hardware de Vetores: {device.upper()}")

embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

# 2. Cliente Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "tdc_index"
qdrant_client = QdrantClient(url=QDRANT_URL)

class SearchRequest(BaseModel):
    text: str
    limit: int = 5

@app.post("/ask")
def search_context(request: SearchRequest):
    start_time = time.time()

    try:
        # A. Gerar Embedding
        query_vector = embeddings_model.embed_query(request.text)

        print(f"\n🔎 Buscando por: '{request.text}'")

        # B. Busca no Qdrant (CORREÇÃO DEFINITIVA: query_points)
        # Substituímos .search() por .query_points() que é mais robusto
        # Retorna um objeto que contém a lista em .points
        search_response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,  # Note: aqui o parametro chama 'query', não 'query_vector'
            limit=request.limit * 2,
            score_threshold=0.6
        )

        # Extraímos a lista de pontos da resposta
        search_result = search_response.points

        # C. Processamento e Deduplicação
        context_text = ""
        sources = []
        seen_ids = set()

        count = 0
        for hit in search_result:
            if count >= request.limit:
                break

            payload = hit.payload
            doc_id = payload.get("mongo_id")

            # Deduplicação
            if doc_id in seen_ids:
                continue

            seen_ids.add(doc_id)
            count += 1

            # Debug Visual
            vector_type = payload.get("vector_type", "unico")
            print(
                f"   🏆 Resultado #{count}: {payload.get('title')} (Via vetor: {vector_type.upper()}) - Score: {hit.score:.3f}")

            # Montagem do Contexto
            if payload.get("type") == "talk":
                context_text += f"---\n{payload.get('page_content')}\n"
                sources.append(payload.get('title'))

            elif payload.get("type") == "info":
                context_text += f"---\n{payload.get('page_content')}\n"
                sources.append("Info Evento")

        elapsed = time.time() - start_time

        # Evita crash se não achar nada
        debug_score = search_result[0].score if search_result else 0

        return {
            "context": context_text,
            "sources": list(set(sources)),
            "debug_score": debug_score,
            "time_taken": elapsed
        }

    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        # Imprime o traceback para ajudar se der erro de novo
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Para rodar: uvicorn search_service:app --reload --port 8000