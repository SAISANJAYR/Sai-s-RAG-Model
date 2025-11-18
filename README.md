# 🤖 Chat-With-Your-Data (RAG Bot)

> **A Retrieval-Augmented Generation (RAG) chatbot built with Gemini 1.5 Flash, LlamaIndex, and ChromaDB.**

## 📖 Overview
This project is a full-stack GenAI application that allows users to upload PDF documents and chat with them in real-time. It overcomes the "knowledge cutoff" and "hallucination" limitations of standard LLMs by using a **RAG pipeline** to ground answers in the user's specific data.

It features a persistent vector database (ChromaDB) to store embeddings, ensuring that documents only need to be indexed once, not every time the app runs.

## 🚀 Features
* **Persistent Memory:** Uses `ChromaDB` to save vector embeddings to disk, allowing for instant startup after the initial ingestion.
* **Source Attribution:** The bot doesn't just answer; it cites its sources, showing exactly which part of the document it referenced.
* **Conversational Memory:** Maintains full context of the chat history, allowing for follow-up questions.
* **Gemini Powered:** Leverages Google's `Gemini 1.5 Flash` for fast, high-quality reasoning and `text-embedding-004` for semantic search.

## 🛠️ Tech Stack
* **LLM:** Google Gemini 1.5 Flash
* **Embeddings:** Google GenAI Embeddings (`models/text-embedding-004`)
* **Orchestration:** LlamaIndex
* **Vector Database:** ChromaDB (Persistent Client)
* **Frontend:** Streamlit

## 🏗️ Architecture
The project follows a two-step pipeline:

```mermaid
graph TD
    A[PDF Document] -->|PyPDF Loader| B(Text Chunks)
    B -->|Gemini Embeddings| C(Vector Embeddings)
    C -->|Store| D[(ChromaDB)]
    
    E[User Question] -->|Gemini Embeddings| F(Query Vector)
    F -->|Semantic Search| D
    D -->|Retrieve Top K Chunks| G[Context]
    
    G -->|Augment Prompt| H[LLM Prompt]
    H -->|Send to Gemini| I[Gemini 1.5 Flash]
    I -->|Response| J[Chat Interface]