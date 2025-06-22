import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch

app = FastAPI()

# Configuración CORRECTA para Azure OpenAI con LangChain
print(f"🔧 Configurando Azure OpenAI...")
print(f"   Base: {os.getenv('OPENAI_API_BASE')}")
print(f"   Version: {os.getenv('OPENAI_API_VERSION')}")

# Configurar variables de entorno para LangChain
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION", "2023-05-15")

# Configurar OpenAI globalmente
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_version = os.getenv("OPENAI_API_VERSION", "2023-05-15")

# Embeddings con configuración Azure específica
embeddings = OpenAIEmbeddings(
    model="demo-embedding",  # Deployment name como model
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type="azure",
    openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-05-15"),
    chunk_size=1
)

# Connect to Azure Cognitive Search
acs = AzureSearch(
    azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
    azure_search_key=os.getenv('SEARCH_API_KEY'),
    index_name=os.getenv('SEARCH_INDEX_NAME'),
    embedding_function=embeddings.embed_query
)

class Body(BaseModel):
    query: str

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)

@app.get('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'openai_base': os.getenv("OPENAI_API_BASE"),
        'openai_version': os.getenv("OPENAI_API_VERSION"),
        'search_service': os.getenv('SEARCH_SERVICE_NAME'),
        'search_index': os.getenv('SEARCH_INDEX_NAME'),
        'env_vars': {
            'OPENAI_API_TYPE': os.environ.get('OPENAI_API_TYPE'),
            'OPENAI_API_BASE': os.environ.get('OPENAI_API_BASE'),
            'OPENAI_API_VERSION': os.environ.get('OPENAI_API_VERSION')
        }
    }

@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    try:
        search_result = search(body.query)
        chat_bot_response = assistant(body.query, search_result)
        return {'response': chat_bot_response}
    except Exception as e:
        return {'error': str(e), 'type': type(e).__name__}

def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    try:
        docs = acs.similarity_search_with_relevance_scores(
            query=query,
            k=5,
        )
        if docs:
            result = docs[0][0].page_content
            print(f"Search result: {result[:100]}...")
            return result
        else:
            return "No relevant documents found"
    except Exception as e:
        print(f"Search error: {e}")
        return f"Search error: {str(e)}"

def assistant(query, context):
    """
    Generate response using Azure OpenAI
    """
    try:
        messages = [
            {"role": "system", "content": "Assistant is a chatbot that helps you find the best wine for your taste."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": context}
        ]

        response = openai.ChatCompletion.create(
            engine="demo-alfredo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"OpenAI error: {e}")
        return f"Error generating response: {str(e)}"