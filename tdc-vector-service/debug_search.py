import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURAÇÃO ---
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "tdc_index"
QUERY = "Rodrigo Tavares"  # Vamos ser diretos no nome

print("🔌 Conectando...")
qdrant_client = QdrantClient(url=QDRANT_URL)

print("📥 Carregando modelo para gerar vetor da pergunta...")
# Importante: O modelo tem que ser IDÊNTICO ao usado no sync
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"🔎 Buscando por: '{QUERY}' no Qdrant...")
query_vector = embeddings_model.embed_query(QUERY)

# Vamos pedir 10 resultados para ver onde o Rodrigo aparece no ranking
hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10
    ).points

print(f"\n🏆 Top Resultados para '{QUERY}':")
print("-" * 50)
for i, hit in enumerate(hits):
    score = hit.score
    title = hit.payload.get('title', 'Sem Título')
    speaker = hit.payload.get('speaker', 'N/A')

    # Destaque visual se for o nosso alvo
    prefix = "✅ ACHOU ->" if "Rodrigo" in speaker else f"#{i + 1}       "

    print(f"{prefix} Score: {score:.4f} | Palestra: {title} | Speaker: {speaker}")

print("-" * 50)