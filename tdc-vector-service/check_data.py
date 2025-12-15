import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"))
COLLECTION = "tdc_index"

print(f"🕵️‍♂️ Investigando se 'Rodrigo Tavares' existe na coleção '{COLLECTION}'...")

# Busca exata pelo campo 'speaker' no payload (Metadado)
# Isso ignora vetores e busca pelo dado cru JSON.
results, _ = client.scroll(
    collection_name=COLLECTION,
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="speaker",
                match=models.MatchValue(value="Rodrigo Tavares")
            )
        ]
    ),
    limit=1,
    with_payload=True,
    with_vectors=False
)

if results:
    point = results[0]
    print("\n✅ ACHOU O REGISTRO!")
    print(f"🆔 ID no Qdrant: {point.id}")
    print(f"📄 Título Salvo: {point.payload.get('title')}")
    print(f"🗣️ Speaker Salvo: {point.payload.get('speaker')}")
    print("-" * 30)
    print("📜 CONTEÚDO VETORIZADO (page_content):")
    print(point.payload.get('page_content'))
    print("-" * 30)

    # Se o page_content estiver vazio ou estranho, achamos o erro.
else:
    print("\n❌ NÃO ENCONTRADO.")
    print("O registro não existe no Qdrant, mesmo que o Sync tenha dito que processou.")
    print("Possível causa: Falha silenciosa no upload ou o nome no 'speaker' ficou salvo diferente.")

    # Vamos listar os 5 primeiros speakers quaisquer para ver como estão salvos
    print("\n📋 Listando 5 speakers aleatórios que ESTÃO no banco:")
    random_points, _ = client.scroll(collection_name=COLLECTION, limit=5)
    for p in random_points:
        print(f" - {p.payload.get('speaker')} (Title: {p.payload.get('title')})")