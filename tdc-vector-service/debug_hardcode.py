import torch
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from numpy.linalg import norm

# 1. Configurar o mesmo modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Rodando teste em: {device.upper()}")

model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

# 2. Definir os textos exatos (Copiado do seu log)
query_text = "Rodrigo Tavares"
doc_text = (
    "ATIVIDADE: Construindo um RAG com seus Próprios Dados do Zero\n"
    "TIPO: TALK | TRILHA: IA Generativa e Dados\n"
    "PALESTRANTE: Rodrigo Tavares (Palestrante)\n"
    "DATA/HORA: Dia 2025-12-10 às 14:30\n"
    "RESUMO: Como montar um RAG com as suas informações do zero...\n"
    "LINKEDIN: https://www.linkedin.com/in/rgtavares/"
)

print("\n🧮 Calculando vetores...")
vec_query = model.embed_query(query_text)
vec_doc = model.embed_query(doc_text) # Usamos embed_query para simular single text

# 3. Cálculo manual de Cosine Similarity
# Fórmula: (A . B) / (||A|| * ||B||)
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

score = cosine_similarity(vec_query, vec_doc)

print("-" * 40)
print(f"🗣️  Query: '{query_text}'")
print(f"📄  Doc: '...{doc_text[:50]}...'")
print("-" * 40)
print(f"🎯 SCORE DE SIMILARIDADE REAL (RAM): {score:.4f}")
print("-" * 40)

if score > 0.5:
    print("✅ CONCLUSÃO: O Modelo funciona! O problema é o upload para o Qdrant (Vetores corrompidos).")
else:
    print("❌ CONCLUSÃO: O Modelo acha que esses textos são diferentes. O prompt/texto está confuso para ele.")