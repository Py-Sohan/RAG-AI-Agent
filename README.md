# ⚙️ RAG AI Agent ⚙️ - Document & Web Content Q&A System

**Created by: Lutfor R. Sohan**

## Overview

![image](https://github.com/user-attachments/assets/8d3cb11c-d34d-4cf4-a627-0ff4f4b64822)


This project implements a powerful **Retrieval-Augmented Generation (RAG) AI Agent** using **Streamlit**, **Langchain**, and **Ollama**. It allows users to interactively chat with their own documents (PDF, TXT, CSV) and web URLs. The agent leverages local LLMs via Ollama and a local vector database (ChromaDB) to provide answers grounded in the provided context, ensuring information relevance and privacy.

This user-friendly interface makes it easy to upload information sources, configure the underlying models, and get answers to specific questions based *only* on the ingested data.

## Features

* **Multi-Source Data Ingestion:** Upload local documents (`.txt`, `.pdf`, `.csv`) or add web URLs directly through the interface.
* **Local LLM Integration:** Utilizes Ollama to run large language models (like Llama 3.2) locally, enhancing privacy and reducing reliance on external APIs.
* **Configurable Model:** Easily change the Ollama model used for embeddings and generation via the sidebar.
* **Persistent Vector Store:** Uses ChromaDB to create and persist a local vector database for efficient document retrieval. The storage path is configurable.
* **Efficient Text Chunking:** Employs `RecursiveCharacterTextSplitter` for optimal document segmentation.
* **Contextual Q&A:** Answers user queries based specifically on the content retrieved from the uploaded documents and URLs.
* **Source Transparency:** Displays the relevant document chunks used to generate the answer, allowing users to verify the information source.
* **Interactive UI:** Built with Streamlit for an intuitive web-based user experience, including file uploading, URL management, and chat interface.

## Technologies Used

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application UI.
* **Langchain:** Framework for orchestrating the RAG pipeline (document loading, splitting, embeddings, retrieval, prompting, LLM interaction).
* **Ollama:** For running local Large Language Models (e.g., `llama3.2`).
* **ChromaDB:** Vector database for storing and retrieving document embeddings locally.
* **Langchain Community/Core Libraries:** Specifically `langchain_community` (loaders), `langchain_ollama` (LLM/Embeddings), `langchain_chroma` (Vector Store), `langchain_core` (prompts).

## Setup & Usage

1.  **Prerequisites:**
    * Ensure you have **Python** (3.8+) installed.
    * Install and run **Ollama** and pull your desired model (e.g., `ollama pull llama3.2`). Make sure the Ollama service is running.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Py-Sohan/RAG-AI-Agent.git
    cd ragn.py
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually:
    # pip install streamlit langchain langchain-community langchain-ollama langchain-chroma beautifulsoup4
    ```
    *(Note: You might need to create a `requirements.txt` file based on the imports in your script)*

4.  **Run the Application:**
    ```bash
    streamlit run your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python file)*

5.  **Using the Agent:**
    * Open the application in your browser (usually `http://localhost:8501`).
    * **(Optional)** Configure the Ollama model name and vector store path in the sidebar if needed.
    * Upload documents (`.txt`, `.pdf`, `.csv`) using the file uploader in the sidebar.
    * Add relevant web page URLs using the URL input in the sidebar.
    * Click "Initialize RAG System". Wait for the documents/URLs to be processed and the vector store to be built.
    * Once initialized, type your questions about the ingested content in the "Ask Questions" section.
    * The agent will retrieve relevant context and generate an answer based solely on that context, displaying both the answer and the source document snippets.

## How It Will Serve Business Purposes?

This RAG AI Agent offers significant value for various business and personal use cases:

* **Internal Knowledge Base Q&A:** Allow employees to quickly find information within internal reports, documentation, policies, or spreadsheets without manual searching.
* **Customer Support:** Augment support staff by providing quick, context-aware answers based on product manuals, FAQs, and past support tickets.
* **Research & Analysis:** Researchers can ingest academic papers, articles, or web pages and ask specific questions to extract key information and insights quickly.
* **Content Summarization & Querying:** Interact with large documents or website content to get summaries or specific answers without reading the entire source.
* **Data Privacy & Control:** By using local LLMs (Ollama) and local vector stores (ChromaDB), sensitive information remains within the user's environment, addressing data privacy concerns.
* **Reduced Hallucination:** The RAG approach grounds the LLM's answers in the provided documents, significantly reducing the likelihood of inaccurate or fabricated information (hallucinations).
