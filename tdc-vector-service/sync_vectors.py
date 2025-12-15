# Permite acessar variáveis de ambiente (MONGO_URI, QDRANT_URL, etc.)
# evitando credenciais hardcoded no código.
import os

# Biblioteca usada para detectar e utilizar GPU (CUDA) ou CPU.
# No nosso caso, o modelo all-MiniLM-L6-v2 roda localmente com aceleração.
import torch

# Cliente oficial do MongoDB, responsável por salvar e ler o conteúdo bruto
# (o nosso “Data Lake” no padrão RAG).
from pymongo import MongoClient

# Cliente do Qdrant, que funciona como nosso banco vetorial de alta performance.
# Ele vai armazenar SOMENTE embeddings + payload leve (IDs).
from qdrant_client import QdrantClient

# Modelos HTTP do Qdrant usados para criar coleções, definir tamanho do vetor,
# função de distância (COSINE) e outras configurações do índice.
from qdrant_client.http import models

# LangChain + HuggingFace:
# Essa camada carrega modelos de embedding de forma padronizada.
# Em vez de lidar com Transformers diretamente, o LangChain simplifica a API
# e fornece métodos convenientes como embed_documents() e embed_query().
from langchain_huggingface import HuggingFaceEmbeddings

# Carrega automaticamente variáveis do arquivo .env para o ambiente.
# É essencial para separar código de configuração (boa prática DevOps).
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Carregamento de variáveis de ambiente (.env)
# -----------------------------------------------------------------------------
# Aqui você pode comentar sobre:
# - separar configuração de código (12-factor app),
# - a mesma app rodando em ambientes diferentes só mudando o .env.
load_dotenv()

# --- 1. SETUP GERAL ----------------------------------------------------------
# Detectamos automaticamente se há GPU disponível (CUDA).
# Isso é ótimo pra demo: você mostra que a mesma aplicação escala de
# "rodar no notebook" até "rodar em servidor com GPU".
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Hardware: {device.upper()}")

# Carregamos o modelo de embedding all-MiniLM-L6-v2.
# Pontos importantes pra você comentar:
# - Ele é open source, leve e rápido.
# - Gera vetores de 384 dimensões.
# - Está rodando LOCAL, dentro do nosso serviço Python (economia de custo e latência).
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

# Constantes de conexão e metadados do índice
MONGO_URI = os.getenv("MONGO_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "tdc_index"
# Tamanho do vetor gerado pelo all-MiniLM-L6-v2.
VECTOR_SIZE = 384

print("🔌 Conectando...")
# Conexão com o nosso "Data Lake" de verdade: MongoDB.
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["tdc_data"]

# Cliente do Qdrant: esse é o nosso "banco vetorial", mas que funciona como ÍNDICE.
qdrant_client = QdrantClient(url=QDRANT_URL)


# --- 2. FORMATAÇÃO ESTRATÉGICA (TEXTOS PARA EMBEDDING) -----------------------
# Aqui entram as funções que transformam JSON em texto "bonito" para o modelo.

def format_event_info(info):
    """
    Função de Pré-Processamento do Evento:

    Pega o JSON bruto do evento (nome, descrição, local, ingressos)
    e transforma em um bloco de texto contínuo, ideal para ser vetorizado.

    Ideia chave pra explicar:
    - O modelo de embedding "entende" texto, não JSON.
    - Então fazemos uma "vista textual" do objeto antes de gerar o vetor.
    """
    prices = ", ".join([f"{t['name']}: {t['price_cash']}" for t in info['tickets']['items']])
    return (
        f"EVENTO: {info['event_name']} ({info['year']})\n"
        f"DESCRIÇÃO: {info['description']}\n"
        f"ONDE: {info['location']['venue']}\n"
        f"INGRESSOS: {prices}"
    )


def generate_dual_vectors(talk):
    """
    🚨 ESTRATÉGIA CHAVE: DOUBLE INDEXING (Indexação Dupla por TEMA e por PESSOA)

    Problema real encontrado:
      - Quando colocamos tudo (nome do palestrante + tema técnico) no MESMO embedding,
        a semântica de "pessoa" e "tópico" se misturam.
      - Perguntar "quem é Rodrigo Tavares?" pode competir com "palestra sobre RAG".

    Solução:
      - Gerar DOIS textos diferentes para a mesma palestra:
        1) Um texto focado em tema (título, trilha, descrição técnica).
        2) Um texto focado em pessoa (nome, bio, cargo, LinkedIn).
      - Cada texto gera um vetor diferente.
      - Ambos os vetores apontam para o mesmo mongo_id.

    Resultado:
      - Se a pergunta falar de tecnologia → o vetor "topic" tende a ganhar.
      - Se a pergunta for pelo nome do palestrante → o vetor "person" tende a ganhar.
    """
    speaker = talk.get('speaker', {})
    name = speaker.get('name', 'Não informado')
    title = talk.get('title')
    track = talk.get('track', 'Geral')

    # 1. CONTEXTO DO TEMA (prioriza vocabulário técnico, trilha, resumo da talk)
    text_topic = (
        f"PALESTRA TÉCNICA: {title}\n"
        f"TRILHA: {track}\n"
        f"RESUMO: {talk.get('description')}\n"
        f"NÍVEL: {talk.get('level', 'Técnico')}"
    )

    # 2. CONTEXTO DO PALESTRANTE (prioriza nome, bio, cargo, LinkedIn)
    text_person = (
        f"QUEM É O PALESTRANTE: {name}\n"
        f"BIO/ROLE: {speaker.get('role', '')}\n"
        f"APRESENTA A PALESTRA: {title}\n"
        f"LINKEDIN: {speaker.get('linkedin', '')}"
    )

    # Retornamos os dois textos + o nome do palestrante (para payload).
    return text_topic, text_person, name


# --- 3. EXECUÇÃO PRINCIPAL ---------------------------------------------------
def main():
    print("🚀 Iniciando Sync com Estratégia 'Double Indexing' (somente índices no Qdrant)...")

    # -------------------------------------------------------------------------
    # 3.1. Limpeza/Recriação da Coleção
    # -------------------------------------------------------------------------
    # Esse passo garante um índice "limpo":
    # - se a coleção já existe, apagamos;
    # - depois criamos de novo com a configuração correta.
    # Em termos de processo, isso é um "full reindex" completo do vetor.
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    # Cria a collection no Qdrant:
    # - VECTOR_SIZE: dimensão do embedding (384).
    # - Distance.COSINE: métrica de similaridade que vamos usar na busca.
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
    )

    # Essas duas listas trabalham em paralelo:
    # - documents_text[i] → texto que vai virar embedding.
    # - payloads[i]       → metadado associado ao vetor gerado a partir daquele texto.
    documents_text = []
    payloads = []

    # -------------------------------------------------------------------------
    # 3.2. Indexação da Info Geral do Evento
    # -------------------------------------------------------------------------
    # Aqui o RAG aprende coisas como:
    # - descrição do TDC,
    # - local,
    # - tipos de ingressos e preços.
    event_info = db.event_info.find_one()
    if event_info:
        # Converte o JSON em texto
        text = format_event_info(event_info)
        # Guarda o texto para virar embedding depois
        documents_text.append(text)

        # E aqui está a parte importante do padrão RAG Enterprise:
        # O payload é LEVE. Guardamos APENAS referência:
        # - mongo_id: para buscar o documento de volta no Mongo.
        # - type: para saber se é info de evento ou talk.
        # - title: só pra fins de debug/exibição.
        payloads.append({
            "mongo_id": str(event_info['_id']),
            "type": "event_info",
            "title": "Informações do Evento"
        })

    # -------------------------------------------------------------------------
    # 3.3. Indexação das Palestras (Double Indexing)
    # -------------------------------------------------------------------------
    # Vamos carregar todas as talks do Mongo (Data Lake).
    talks = list(db.talks.find({}))
    print(f"📦 Processando {len(talks)} palestras (gerando 2 vetores por palestra)...")

    for talk in talks:
        # Gera os dois textos: um para o tema, outro para a pessoa.
        txt_topic, txt_person, speaker_name = generate_dual_vectors(talk)

        # Payload base com o ID do Mongo.
        # ESSENCIAL: é esse ID que será usado depois para "hidratar" o contexto
        # indo buscar o texto completo no Mongo na hora da pergunta.
        base_payload = {
            "mongo_id": str(talk['_id']),
            "type": "talk",
            "title": talk.get('title'),
            "speaker": speaker_name
        }

        # VETOR 1: TEMA (vector_type = topic)
        documents_text.append(txt_topic)
        payload1 = base_payload.copy()
        payload1["vector_type"] = "topic"  # útil para debug e análise de relevância
        payloads.append(payload1)

        # VETOR 2: PESSOA (vector_type = person)
        documents_text.append(txt_person)
        payload2 = base_payload.copy()
        payload2["vector_type"] = "person"  # idem: ajuda a entender de onde veio o match
        payloads.append(payload2)

    # -------------------------------------------------------------------------
    # 3.4. Geração de Embeddings em Lote
    # -------------------------------------------------------------------------
    # Agora, com todos os textos consolidados, mandamos gerar os vetores.
    # Isso é mais eficiente do que chamar o modelo um por um.
    print(f"🧠 Gerando Embeddings para {len(documents_text)} vetores em {device.upper()}...")
    vectors = embeddings_model.embed_documents(documents_text)

    # -------------------------------------------------------------------------
    # 3.5. Upload para o Qdrant (Índice Vetorial)
    # -------------------------------------------------------------------------
    # Aqui acontece a "mágica" da indexação:
    # - Cada vetor gerado vai para o Qdrant,
    # - Acompanhado do payload leve (com mongo_id, type, title, vector_type...).
    # Importante reforçar na fala:
    #   ❌ NÃO estamos salvando o texto completo no Qdrant.
    #   ✅ SOMENTE embeddings + IDs → Mongo continua sendo a fonte de verdade.
    print(f"💾 Salvando {len(vectors)} pontos no Qdrant (somente índices, sem conteúdo bruto)...")
    qdrant_client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payloads
    )

    print("✅ Sincronização Finalizada! Qdrant agora guarda apenas embeddings + IDs (Mongo como fonte de verdade).")


# Ponto de entrada do script.
# Na narrativa de arquitetura:
# - isso aqui poderia ser um job agendado (cron, Azure Functions Timer Trigger,
#   GitHub Actions, Azure DevOps Pipeline, etc.).
if __name__ == "__main__":
    main()
