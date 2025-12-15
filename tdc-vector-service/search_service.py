# Módulo padrão do Python para acessar variáveis de ambiente (ex: MONGO_URI, QDRANT_URL, etc.)
import os

# Usado para medir tempos de execução (latência de embedding, Qdrant, Mongo, GPT, etc.)
import time

# Biblioteca para detectar e usar GPU (CUDA) ou CPU ao rodar o modelo de embeddings
import torch

# Framework web assíncrono que expõe nossa API HTTP (/ask, /debug/search)
from fastapi import FastAPI

# Middleware para habilitar CORS (permitir que o frontend em outro domínio/porta acesse essa API)
from fastapi.middleware.cors import CORSMiddleware

# Base para definir modelos de request/response tipados (SearchRequest, SearchResponse, etc.)
from pydantic import BaseModel

# Cliente oficial do MongoDB em Python, usado para conectar e consultar o nosso “Data Lake”
from pymongo import MongoClient

# Classe para trabalhar com IDs do Mongo (_id), convertendo strings em ObjectId e vice-versa
from bson.objectid import ObjectId

# Cliente do Qdrant, o banco vetorial usado como índice de similaridade semântica
from qdrant_client import QdrantClient

# Wrapper do LangChain para carregar o modelo de embeddings HuggingFace (all-MiniLM-L6-v2)
from langchain_huggingface import HuggingFaceEmbeddings

# Wrapper do LangChain para se conectar ao Azure OpenAI (GPT) via deployment configurado
from langchain_openai import AzureChatOpenAI

# Tipos de mensagens usados para montar o prompt de chat (System + Human) para o GPT
from langchain_core.messages import SystemMessage, HumanMessage

# Facilita o carregamento de variáveis de ambiente a partir de um arquivo .env
from dotenv import load_dotenv

# =============================================================================
# 1. CONFIGURAÇÃO INICIAL
# =============================================================================
# Esse arquivo é essencialmente o "orquestrador" do RAG:
# - recebe a pergunta do frontend
# - vetoriza localmente
# - busca no Qdrant
# - hidrata no Mongo
# - monta contexto
# - chama Azure OpenAI
# - devolve resposta e fontes para o frontend
# Tudo isso numa API HTTP simples, via FastAPI.

load_dotenv()

app = FastAPI(title="TDC RAG Search Service", version="1.1.0")

# -------------------------------------------------------------------------
# CORS liberado (para demo)
# -------------------------------------------------------------------------
# Durante a palestra / protótipo, é muito mais simples liberar tudo.
# Em produção:
#   - aqui você restringiria domains específicos do front
#   - controlaria métodos e headers aceita dos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # em prod → ["https://meu-front.com", ...]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Hardware para embeddings
# -------------------------------------------------------------------------
# Mesma lógica dos outros scripts:
# - se tiver GPU (CUDA), usa GPU
# - se não tiver, cai pra CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 API Iniciada | Hardware de Vetorização: {device.upper()}")

# -------------------------------------------------------------------------
# Modelo de embeddings local (all-MiniLM-L6-v2)
# -------------------------------------------------------------------------
# Esse é o "cérebro semântico" da parte de busca vetorial.
# Ponto importante pra explicar:
# - Não estamos usando GPT para embeddings aqui.
# - O modelo é local, open source, rápido e barato.
print("📥 Carregando modelo de vetores (Local all-MiniLM-L6-v2)...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

# -------------------------------------------------------------------------
# Cliente do Azure OpenAI (Texto → Resposta final)
# -------------------------------------------------------------------------
# Esse é o modelo "grande" (GPT) que:
# - lê o contexto vindo do Mongo
# - responde a pergunta do usuário
#
# Observação importante:
# - Não setamos temperature porque esse deployment específico
#   não aceita override (erro 400 se fizer isso).
print(f"🤖 Conectando ao Azure OpenAI ({os.getenv('AZURE_DEPLOYMENT_NAME')})...")
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    max_tokens=800,
    # importante: não setar temperature aqui, o modelo do deployment não aceita
)

# -------------------------------------------------------------------------
# MongoDB: Data Lake / Fonte de Verdade
# -------------------------------------------------------------------------
# - Aqui estão os documentos COMPLETOS.
# - Toda a hidratação do RAG acontece consultando esse banco.
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["tdc_data"]

# -------------------------------------------------------------------------
# Qdrant: Índice Vetorial
# -------------------------------------------------------------------------
# - Aqui só vivem embeddings + payload leve (IDs).
# - Servidor de similaridade semântica.
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
COLLECTION_INDEX = "tdc_index"


# =============================================================================
# 2. MODELOS Pydantic (Request/Response)
# =============================================================================

class SearchRequest(BaseModel):
    """
    Payload de entrada padrão da API:
    - text: pergunta do usuário
    - limit: quantos documentos únicos (do Mongo) queremos para montar o contexto
    """
    text: str
    limit: int = 5


class SearchResponse(BaseModel):
    """
    Resposta final para o frontend:
    - answer: texto gerado pelo GPT (já em markdown)
    - sources: lista de títulos/fontes que alimentaram o contexto
    - time_taken: tempo total da operação
    - timings: tempos detalhados por etapa (embedding, qdrant, mongo, gpt)
    """
    answer: str
    sources: list[str]
    time_taken: float
    timings: dict[str, float] | None = None  # tempos por etapa em milissegundos


class DebugHit(BaseModel):
    """
    Modelo de cada hit para o endpoint de debug:
    - mongo_id: referência pro documento real no Mongo
    - title: título da talk ou nome do evento
    - type: talk ou event_info
    - vector_type: topic/person/single (útil pra explicar o Double Indexing)
    - score: score de similaridade retornado pelo Qdrant
    """
    mongo_id: str
    title: str | None = None
    type: str | None = None
    vector_type: str | None = None
    score: float


class DebugSearchResponse(BaseModel):
    """
    Resposta do endpoint /debug/search:
    - query: pergunta original
    - limit: limite pedido
    - hits: lista deduplicada de resultados do Qdrant
    """
    query: str
    limit: int
    hits: list[DebugHit]


# =============================================================================
# 3. FUNÇÃO AUXILIAR: CONSTRUÇÃO DO CONTEXTO
# =============================================================================

def build_context(docs):
    """
    Monta um contexto legível a partir dos documentos do Mongo.

    Ponto central do RAG Enterprise:
    - O contexto é sempre montado a partir do Data Lake (MongoDB),
      nunca do texto armazenado no Qdrant.
    - O Qdrant é só um índice para achar quais documentos são relevantes.

    Essa função:
    - formata dados de event_info e talks
    - gera um "textão" bem estruturado pro GPT consumir.
    """
    context = ""
    for doc in docs:
        # Caso: documento de informações gerais do evento
        if "event_name" in doc:
            context += f"--- DADOS GERAIS DO EVENTO ---\n"
            context += f"Evento: {doc['event_name']} ({doc.get('year', '')})\n"

            location = doc.get("location", {})
            context += f"Local: {location.get('venue', '')} ({location.get('address', '')})\n"

            tickets = doc.get("tickets", {}).get("items", [])
            prices = ", ".join([f"{i['name']}: {i['price_cash']}" for i in tickets])
            context += f"Ingressos: {prices}\n"

            policies = doc.get("policies", {})
            if "cancellation" in policies:
                context += f"Política de cancelamento: {policies['cancellation']}\n"
            context += "\n"

        # Caso: documento de talk (palestra/atividade)
        elif "title" in doc:
            speaker = doc.get("speaker", {})
            context += f"--- ATIVIDADE ---\n"
            context += f"Título: {doc['title']}\n"
            context += f"Tipo: {doc.get('type', '').upper()} | Trilha: {doc.get('track', '')}\n"
            context += f"Palestrante: {speaker.get('name', 'Não informado')}\n"
            context += f"Data/Hora: {doc.get('date', '')} às {doc.get('time', '')}\n"
            context += f"Descrição: {doc.get('description', '')}\n\n"

    return context


# =============================================================================
# 4. ENDPOINT DE DEBUG: /debug/search (SEM GPT, SEM MONGO)
# =============================================================================

@app.post("/debug/search", response_model=DebugSearchResponse)
async def debug_search(request: SearchRequest):
    """
    Endpoint para inspecionar diretamente o resultado do Qdrant:

    O que ele faz:
    - Vetoriza a query.
    - Consulta o Qdrant.
    - Deduplica por mongo_id.
    - Retorna apenas infos básicas (sem chamar Mongo, sem GPT).

    Por que isso é ótimo pra palestra:
    - Dá pra mostrar ao vivo:
      - os scores,
      - o tipo do documento,
      - se veio por vetor "topic" ou "person",
      - como o Double Indexing está se comportando.
    """
    print(f"\n🧪 [DEBUG] Buscando no Qdrant para a query: {request.text!r}")

    # 1) Medição do tempo de embedding
    t0 = time.perf_counter()
    query_vector = embeddings_model.embed_query(request.text)
    t1 = time.perf_counter()

    # 2) Busca no Qdrant com um limite um pouco maior (para deduplicar depois)
    qdrant_limit = request.limit * 2
    search = qdrant_client.query_points(
        collection_name=COLLECTION_INDEX,
        query=query_vector,
        limit=qdrant_limit,
        score_threshold=0.0  # sem filtro de score, você vê tudo e decide no front
    )
    t2 = time.perf_counter()

    hits_model: list[DebugHit] = []
    seen_ids: set[str] = set()

    # 3) Deduplicação por mongo_id
    for hit in search.points:
        payload = hit.payload or {}
        mongo_id = payload.get("mongo_id")
        if not mongo_id:
            continue

        if mongo_id in seen_ids:
            continue
        seen_ids.add(mongo_id)

        # Constrói o modelo de resposta amigável pro front / demo
        hits_model.append(
            DebugHit(
                mongo_id=mongo_id,
                title=payload.get("title"),
                type=payload.get("type"),
                vector_type=payload.get("vector_type"),
                score=hit.score,
            )
        )

        if len(hits_model) >= request.limit:
            break

    embed_ms = (t1 - t0) * 1000
    qdrant_ms = (t2 - t1) * 1000
    print(f"⏱️ [DEBUG] embedding={embed_ms:.2f}ms | qdrant={qdrant_ms:.2f}ms | hits={len(hits_model)}")

    return DebugSearchResponse(
        query=request.text,
        limit=request.limit,
        hits=hits_model,
    )


# =============================================================================
# 5. ENDPOINT PRINCIPAL: /ask (FLUXO COMPLETO DE RAG)
# =============================================================================

@app.post("/ask", response_model=SearchResponse)
async def ask_endpoint(request: SearchRequest):
    """
    Fluxo completo de RAG:

    1. Vetoriza a pergunta (modelo local all-MiniLM-L6-v2).
    2. Busca semelhante no Qdrant (traz só IDs + payload leve).
    3. Deduplica IDs e separa por tipo (talk vs event_info).
    4. Hidrata dados completos no MongoDB.
    5. Monta o contexto em texto.
    6. Chama Azure OpenAI (GPT) com contexto + pergunta.
    7. Retorna resposta final + fontes + tempos detalhados.
    """
    start = time.perf_counter()
    query = request.text
    print(f"\n💬 Pergunta: {query}")

    timings: dict[str, float] = {}

    # ---------------------------------------------------------------------
    # 1) Vetorização da pergunta
    # ---------------------------------------------------------------------
    t0 = time.perf_counter()
    query_vector = embeddings_model.embed_query(query)
    t1 = time.perf_counter()
    timings["embedding_ms"] = (t1 - t0) * 1000

    # ---------------------------------------------------------------------
    # 2) Busca no Qdrant (apenas IDs + metadados)
    # ---------------------------------------------------------------------
    # Multiplica o limit por 2 pra dar espaço para deduplicação.
    qdrant_limit = request.limit * 2
    t2 = time.perf_counter()
    search = qdrant_client.query_points(
        collection_name=COLLECTION_INDEX,
        query=query_vector,
        limit=qdrant_limit,
        score_threshold=0.5  # threshold ajustável; mais alto = mais estrito
    )
    t3 = time.perf_counter()
    timings["qdrant_ms"] = (t3 - t2) * 1000

    hits = search.points

    if not hits:
        total = (time.perf_counter() - start) * 1000
        print("⚠️ Nenhum resultado vetorial encontrado no Qdrant.")
        return SearchResponse(
            answer="Não encontrei informações relevantes no índice para responder sua pergunta.",
            sources=[],
            time_taken=total / 1000.0,
            timings=timings
        )

    # ---------------------------------------------------------------------
    # 3) Deduplicação de IDs e separação por tipo
    # ---------------------------------------------------------------------
    seen_ids: set[str] = set()
    talk_ids: list[ObjectId] = []
    info_ids: list[ObjectId] = []

    print("🧠 Resultados do Qdrant (antes da deduplicação):")
    for hit in hits:
        payload = hit.payload or {}
        mongo_id = payload.get("mongo_id")
        if not mongo_id:
            continue

        # evita usar o mesmo documento duas vezes no contexto
        if mongo_id in seen_ids:
            continue

        seen_ids.add(mongo_id)

        p_type = payload.get("type")
        vector_type = payload.get("vector_type", "single")

        # Log amigável pra você mostrar no terminal durante a demo
        print(
            f"   • {payload.get('title', 'Sem título')} "
            f"(type={p_type}, via vetor={vector_type}, score={hit.score:.3f})"
        )

        oid = ObjectId(mongo_id)
        if p_type == "event_info":
            info_ids.append(oid)
        elif p_type == "talk":
            talk_ids.append(oid)

        # respeita o limite de documentos únicos para o contexto
        if len(seen_ids) >= request.limit:
            break

    # ---------------------------------------------------------------------
    # 4) Hidratação no MongoDB
    # ---------------------------------------------------------------------
    # Agora, com os IDs na mão, buscamos o conteúdo completo no Mongo.
    t4 = time.perf_counter()
    docs = []

    if info_ids:
        docs.extend(list(db.event_info.find({"_id": {"$in": info_ids}})))

    if talk_ids:
        docs.extend(list(db.talks.find({"_id": {"$in": talk_ids}})))

    t5 = time.perf_counter()
    timings["mongo_ms"] = (t5 - t4) * 1000

    if not docs:
        total = (time.perf_counter() - start) * 1000
        print("⚠️ Qdrant retornou IDs, mas não encontrei documentos no Mongo.")
        return SearchResponse(
            answer="Não consegui localizar os detalhes dessas informações no banco de dados.",
            sources=[],
            time_taken=total / 1000.0,
            timings=timings
        )

    # ---------------------------------------------------------------------
    # 5) Construção do contexto em texto
    # ---------------------------------------------------------------------
    context_str = build_context(docs)

    # Prompt de sistema:
    # - define o papel do modelo
    # - reforça o idioma
    # - reforça a regra de usar só o contexto
    system_prompt = """
    INSTRUCTIONS
    ===
    - Você é o Assistente Oficial do TDC Experience.
    - MUITO IMPORTANTE:
      - Responda SEMPRE no mesmo idioma da pergunta, de forma clara e direta.
      - Se a pergunta estiver em outro idioma que não seja Português,
        traduza mentalmente as informações do evento para o mesmo idioma antes de responder.
    - Use APENAS o contexto fornecido abaixo. Se a informação não estiver no contexto,
      diga que não possui dados suficientes para responder.
    - Use o formato Markdown para as respostas.
    """

    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"CONTEXTO RECUPERADO:\n{context_str}"),
        HumanMessage(content=query)
    ]

    # ---------------------------------------------------------------------
    # 6) Geração com Azure OpenAI (GPT)
    # ---------------------------------------------------------------------
    print("🤖 Gerando resposta com Azure OpenAI...")
    t6 = time.perf_counter()
    ai_response = llm.invoke(messages)
    t7 = time.perf_counter()
    timings["gpt_ms"] = (t7 - t6) * 1000

    total = (time.perf_counter() - start) * 1000
    print(
        f"✅ Resposta gerada em {total:.2f}ms | "
        f"embedding={timings['embedding_ms']:.2f}ms | "
        f"qdrant={timings['qdrant_ms']:.2f}ms | "
        f"mongo={timings['mongo_ms']:.2f}ms | "
        f"gpt={timings['gpt_ms']:.2f}ms"
    )

    # ---------------------------------------------------------------------
    # 7) Fontes (para exibir no frontend)
    # ---------------------------------------------------------------------
    # Aqui você extrai nomes amigáveis dos documentos usados:
    # - título das talks
    # - nome do evento
    sources: list[str] = []
    for d in docs:
        if "title" in d:
            sources.append(d["title"])
        elif "event_name" in d:
            sources.append(d["event_name"])
        else:
            sources.append("Fonte desconhecida")

    return SearchResponse(
        answer=ai_response.content,
        sources=sources,
        time_taken=total / 1000.0,
        timings=timings
    )
