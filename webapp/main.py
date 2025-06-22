import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch

app = FastAPI()

# Configuración de OpenAI Azure - Versión actualizada
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_version = "2023-05-15"  # Cambiar a versión compatible

# Configuración de embeddings con parámetros adicionales
embeddings = OpenAIEmbeddings(
    deployment="demo-embedding",
    model="text-embedding-ada-002",
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_type="azure",
    openai_api_version="2023-05-15",
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
        'search_service': os.getenv('SEARCH_SERVICE_NAME'),
        'search_index': os.getenv('SEARCH_INDEX_NAME')
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