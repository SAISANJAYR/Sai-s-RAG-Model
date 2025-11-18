import streamlit as st
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

load_dotenv() # Load your GOOGLE_API_KEY

# --- This is the key LlamaIndex step ---
# Configure the global Settings
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model="text-embedding-004")

st.title("Final RAG App")

# --- Load the persistent index from ChromaDB ---
@st.cache_resource(show_spinner="Loading index from ChromaDB...")
def load_from_chroma():
    db = chromadb.PersistentClient(path="D:\RAG Model\llama-gemini-rag\chroma_db")
    chroma_collection = db.get_collection("my_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

# ... (at the top of app.py)
index = load_from_chroma()

# --- Use st.session_state to store the chat engine ---
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context")

# --- Store and display chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input logic ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ... (inside the "if prompt" block)
    with st.chat_message("assistant"):
        response = st.session_state.chat_engine.chat(prompt)
        # ... (get response) ...
        st.markdown(response.response)

        # --- Add the source expander ---
        with st.expander("See Sources"):
            for node in response.source_nodes:
                st.write(f"- **Source:** {node.metadata.get('file_name', 'N/A')}")
                st.text(node.get_content()[:500] + "...") # Show snippet

        st.session_state.messages.append({"role": "assistant", "content": response.response})

if st.button("Clear History"): 
    st.session_state.messages = []