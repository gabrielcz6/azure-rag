import os
import openai
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv('.env')

print("🔧 Configurando OpenAI...")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_version = "2023-05-15"

print("🔧 Configurando embeddings...")
embeddings = OpenAIEmbeddings(deployment="demo-embedding", chunk_size=1)

print("🔧 Conectando a Azure Cognitive Search...")
acs = AzureSearch(
    azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
    azure_search_key=os.getenv('SEARCH_API_KEY'),
    index_name=os.getenv('SEARCH_INDEX_NAME'),
    embedding_function=embeddings.embed_query
)

# Buscar el archivo CSV
csv_paths = [
    "wine-ratings.csv",
    "examples/1-setup-application/wine-ratings.csv"
]

csv_file = None
for path in csv_paths:
    if os.path.exists(path):
        csv_file = path
        print(f"📄 Archivo CSV encontrado: {path}")
        break

if not csv_file:
    print("❌ Error: No se encontró wine-ratings.csv")
    exit()

print("📚 Cargando archivo CSV...")
loader = CSVLoader(csv_file, encoding='utf-8')
try:
    documents = loader.load()
    print(f"✅ Total de documentos en CSV: {len(documents)}")
except UnicodeDecodeError:
    print("⚠️ Error de UTF-8, intentando con latin1...")
    loader = CSVLoader(csv_file, encoding='latin1')
    documents = loader.load()
    print(f"✅ Total de documentos en CSV: {len(documents)}")

# 🎯 OPTIMIZACIÓN EXTREMA PARA S0: Solo muestra muy pequeña
MAX_DOCS = 10  # Solo 10 documentos para tier S0
print(f"🔥 TIER S0 DETECTADO: Procesando solo {MAX_DOCS} documentos de {len(documents)}")
print("📊 Límites S0: 20 RPM (~0.33 requests/second)")
print("⏰ Esto tomará aproximadamente 3-5 minutos con delays seguros")

if len(documents) > MAX_DOCS:
    documents = documents[:MAX_DOCS]

print("✂️ Dividiendo documentos...")
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)  # Chunks más pequeños
docs = text_splitter.split_documents(documents)
print(f"✅ Procesando {len(docs)} chunks")

print("🚀 Creando índice - MODO SÚPER CONSERVADOR...")

# 🛡️ ESTRATEGIA ULTRA CONSERVADORA PARA S0
REQUEST_DELAY = 4.0  # 4 segundos entre requests (muy conservador)
BATCH_SIZE = 1       # 1 documento por vez
total_requests = len(docs)

print(f"⏰ Tiempo estimado: {total_requests * REQUEST_DELAY / 60:.1f} minutos")
print("🔄 Procesando documento por documento...")

successful_docs = 0
failed_docs = 0

try:
    for i, doc in enumerate(docs):
        doc_num = i + 1
        print(f"📦 Procesando documento {doc_num}/{len(docs)}...")
        
        try:
            # Agregar UN documento por vez
            acs.add_documents(documents=[doc])
            successful_docs += 1
            print(f"✅ Documento {doc_num} completado")
            
            # Delay fijo entre cada request
            if doc_num < len(docs):  # No delay después del último
                print(f"⏳ Esperando {REQUEST_DELAY}s (rate limit S0)...")
                time.sleep(REQUEST_DELAY)
                
        except Exception as doc_error:
            failed_docs += 1
            print(f"⚠️ Error en documento {doc_num}: {str(doc_error)[:100]}...")
            
            # Si hay rate limit, esperar más tiempo
            if "rate limit" in str(doc_error).lower() or "429" in str(doc_error):
                print("🛑 Rate limit detectado. Esperando 60 segundos...")
                time.sleep(60)
            else:
                print("⏳ Esperando 10 segundos antes de continuar...")
                time.sleep(10)
    
    print(f"\n🎉 Proceso completado!")
    print(f"✅ Documentos exitosos: {successful_docs}")
    print(f"❌ Documentos fallidos: {failed_docs}")
    
except KeyboardInterrupt:
    print(f"\n⚠️ Proceso cancelado por usuario")
    print(f"✅ Documentos procesados antes de cancelar: {successful_docs}")
except Exception as e:
    print(f"❌ Error general: {e}")

# Solo probar búsqueda si se procesó al menos 1 documento
if successful_docs > 0:
    print("\n🔍 Probando búsqueda...")
    try:
        results = acs.similarity_search_with_relevance_scores(
            query="best wine",
            k=1,
        )
        print(f"✅ Encontrados {len(results)} resultados")
        if results:
            content = results[0][0].page_content[:100]
            relevance = results[0][1]
            print(f"🍷 Resultado: {content}...")
            print(f"🎯 Relevancia: {relevance:.3f}")
    except Exception as e:
        print(f"⚠️ Error en búsqueda: {e}")

print(f"\n📊 Resumen final:")
print(f"   🎯 Documentos objetivo: {MAX_DOCS} (optimizado para S0)")
print(f"   ✅ Documentos indexados: {successful_docs}")
print(f"   🔍 Índice: {os.getenv('SEARCH_INDEX_NAME')}")
print(f"   🌐 Servicio: {os.getenv('SEARCH_SERVICE_NAME')}")

if successful_docs > 0:
    print(f"\n🚀 ¡Índice básico creado! Ya puedes:")
    print(f"   - Ejecutar tu aplicación FastAPI")
    print(f"   - Hacer deploy a Azure")
    print(f"   - Agregar más documentos después si necesitas")
else:
    print(f"\n❌ No se creó el índice. Revisa:")
    print(f"   - Que las keys de OpenAI sean correctas")
    print(f"   - Que el deployment 'demo-embedding' exista")
    print(f"   - Considera solicitar aumento de quota: https://aka.ms/oai/quotaincrease")