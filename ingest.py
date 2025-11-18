import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
import os
from dotenv import load_dotenv

load_dotenv()

# Configure global settings
Settings.llm = None # We don't need the LLM for ingestion
Settings.embed_model = GoogleGenAIEmbedding(model="text-embedding-004")

print("Starting ingestion...")

# 1. Init ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("my_docs")

# 2. Create the LlamaIndex vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Create the StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Load documents
reader = SimpleDirectoryReader("D:\RAG Model\llama-gemini-rag\data")
documents = reader.load_data()

# 5. Load and ingest (this populates the ChromaDB collection)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print("Ingestion complete. You can now run the app.py")