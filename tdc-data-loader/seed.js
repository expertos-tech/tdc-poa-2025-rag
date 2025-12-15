// seed.js
// ============================================================================
// Este script representa a **primeira etapa do pipeline de RAG**:
// aqui nós criamos o "Data Lake" da aplicação, carregando os dados brutos
// (palestras + informações do evento) dentro do MongoDB.
// ============================================================================

// Importamos o cliente nativo do MongoDB.
// Na narrativa da palestra, aqui você pode falar de:
// - "camada de dados corporativa"
// - "fonte única da verdade"
// - "Lake / Lakehouse onde o texto completo fica armazenado".
const { MongoClient } = require('mongodb');

// --------------------------------------------------------------------------
// Importação dos dados brutos
// --------------------------------------------------------------------------
// Esses arquivos JS simulam a "fonte oficial" de dados do TDC:
// poderiam ser um JSON vindo de uma API, um CSV, um HTML raspado, etc.
// Na prática, é a **origem do conhecimento** que o RAG vai usar.
const talksDay1 = require('./data/talks_day1');
const talksDay2 = require('./data/talks_day2');
const eventInfo = require('./data/event_info');

// --------------------------------------------------------------------------
// Configuração da conexão com o MongoDB
// --------------------------------------------------------------------------
// Aqui definimos a URI do MongoDB.
// - No ambiente local/Docker: usuário/senha admin/admin.
// - authSource=admin indica onde as credenciais serão validadas.
// Em PRO, isso seria uma connection string segura (Key Vault, variável de ambiente,
// managed identity, etc.).
const uri = "mongodb://admin:admin@localhost:27017/?authSource=admin";

// Criamos uma instância do cliente Mongo.
// Ele ainda NÃO está conectado; isso só acontece quando chamamos client.connect().
const client = new MongoClient(uri);

// --------------------------------------------------------------------------
// Função principal de carga
// --------------------------------------------------------------------------
// Usamos uma função async para:
// - conectar no banco,
// - limpar dados antigos,
// - inserir os novos documentos,
// - e fechar conexão no final.
// Isso simula um "job de ingestão" que poderia rodar em batch (cron, pipeline, etc.).
async function run() {
  try {
    console.log("🔌 Conectando ao MongoDB...");
    // Abre a conexão física com o servidor Mongo.
    // Aqui você pode comentar sobre pool de conexões, latência, etc.
    await client.connect();
    
    // Escolhe (ou cria se não existir) o database onde vamos trabalhar.
    // No contexto da palestra, esse DB é o nosso "Data Lake lógico".
    const db = client.db("tdc_data");
    console.log("✅ Conectado ao banco 'tdc_data'");

    // ----------------------------------------------------------------------
    // 1. LIMPEZA (Reset do Data Lake)
    // ----------------------------------------------------------------------
    // Antes de uma nova sincronização, limpamos as coleções.
    // Isso demonstra o conceito de "reindexação" ou "full reload":
    // apagamos tudo e recarregamos a versão mais atual da fonte oficial.
    console.log("🧹 Limpando coleções antigas...");
    await db.collection("talks").deleteMany({});
    await db.collection("event_info").deleteMany({});

    // Aqui você pode explicar:
    // - Em produção, às vezes fazemos "upsert" em vez de apagar tudo.
    // - Mas para uma demo de RAG, é didático mostrar um full refresh.

    // ----------------------------------------------------------------------
    // 2. CONSOLIDAÇÃO DOS DADOS
    // ----------------------------------------------------------------------
    // Junta todas as atividades (palestras, keynotes, etc.) em um único array.
    // Isso é o "conteúdo bruto" que o modelo vai ler DEPOIS que o Qdrant
    // devolver os IDs.
    const allTalks = [...talksDay1, ...talksDay2];

    // ----------------------------------------------------------------------
    // 3. INSERÇÃO NO MONGODB (Data Lake)
    // ----------------------------------------------------------------------
    // Esse é o ponto-chave da narrativa:
    // - Aqui nós guardamos o **TEXTO COMPLETO** (título, descrição, palestrante...)
    // - O MongoDB vira a **Fonte de Verdade**.
    // - O Qdrant NÃO armazena o texto completo, ele só guarda um índice vetorial
    //   que aponta para o _id desses documentos.
    console.log(`🚀 Inserindo ${allTalks.length} palestras/atividades...`);

    // insertMany grava todas as palestras de uma vez.
    // O retorno traz, por exemplo, o número de documentos inseridos.
    const talksResult = await db.collection("talks").insertMany(allTalks);

    console.log(`📝 ${talksResult.insertedCount} palestras inseridas.`);

    // Além das talks, também carregamos uma coleção com as informações gerais
    // do evento (local, descrição, preços, políticas, etc).
    // Isso permite que o RAG responda perguntas do tipo:
    // - "Onde vai ser o TDC Porto Alegre?"
    // - "Quais são os tipos de ingresso?"
    console.log("🚀 Inserindo informações do evento...");
    await db.collection("event_info").insertOne(eventInfo);
    console.log("📝 Informações do evento inseridas.");

    console.log("✨ Carga de dados finalizada com sucesso!");
    // Aqui você pode fazer uma pausa e reforçar:
    // - "Neste momento, o nosso Data Lake está pronto."
    // - "Nada de vetor ainda. Só dados brutos no Mongo."
    // - "O próximo passo é indexar isso em forma de embeddings no Qdrant."

  } catch (error) {
    // Bloco de tratamento de erro:
    // Qualquer problema de conexão/inserção cai aqui.
    // Em produção, você poderia logar em um sistema centralizado (App Insights, Datadog, etc.).
    console.error("❌ Erro na carga de dados:", error);
  } finally {
    // O finally SEMPRE é executado, com sucesso ou erro.
    // Fechar a conexão explicitamente é uma boa prática em scripts one-shot.
    await client.close();
    console.log("👋 Conexão fechada.");
  }
}

// Chamamos a função principal.
// Na palestra, isso te dá gancho para falar de:
// - scripts de seed rodando em CI/CD,
// - jobs agendados (cron, Azure Functions Timer, etc.),
// - ou pipelines de ingestão (Data Factory / Synapse / Airflow).
run();
