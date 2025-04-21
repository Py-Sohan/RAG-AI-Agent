import streamlit as st
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Configuration
os.environ['USER_AGENT'] = 'CarbonRAG/1.0'
SUPPORTED_EXTENSIONS = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.csv': CSVLoader,
}

def main():
    st.set_page_config(page_title="Document RAG System", layout="wide")
    st.title("⚙️ RAG AI Agent ⚙️")
    st.text("Created by Lutfor R. Sohan")

    # Initialize session state
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.vector_db_path = "./vector_db"
        st.session_state.llm_model = "llama3.2"
        st.session_state.urls = []

    # Document upload section
    with st.sidebar:
        st.header("Configuration")
        st.session_state.vector_db_path = st.text_input(
            "Vector store path",
            st.session_state.vector_db_path
        )
        st.session_state.llm_model = st.text_input(
            "Ollama model name",
            st.session_state.llm_model
        )

        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=list(SUPPORTED_EXTENSIONS.keys()),
            accept_multiple_files=True
        )

        st.header("Add URLs")
        new_url = st.text_input("Enter a URL")
        if st.button("Add URL"):
            if new_url:
                st.session_state.urls.append(new_url)
                st.rerun()

        st.write("### URLs Added:")
        urls_to_remove = []
        for i, url in enumerate(st.session_state.urls):
            col1, col2 = st.columns([8, 1])
            col1.write(f"- {url}")
            if col2.button("❌", key=f"remove_{i}"):
                urls_to_remove.append(i)

        for i in sorted(urls_to_remove, reverse=True):
            st.session_state.urls.pop(i)

        if urls_to_remove:
            st.rerun()

        if st.button("Initialize RAG System"):
            if not uploaded_files and not st.session_state.urls:
                st.error("Please upload files or enter URLs")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        # Process uploaded files
                        docs = []
                        for uploaded_file in uploaded_files:
                            ext = os.path.splitext(uploaded_file.name)[1].lower()
                            if ext in SUPPORTED_EXTENSIONS:
                                try:
                                    # Save temporarily to load
                                    temp_path = f"./temp_{uploaded_file.name}"
                                    with open(temp_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

                                    loader = SUPPORTED_EXTENSIONS[ext](temp_path)
                                    docs.extend(loader.load())
                                    os.remove(temp_path)
                                    st.success(f"Loaded: {uploaded_file.name}")
                                except Exception as e:
                                    st.error(f"Failed to load {uploaded_file.name}: {str(e)}")

                        # Process web URLs
                        for url in st.session_state.urls:
                            try:
                                docs.extend(WebBaseLoader(url).load())
                                st.success(f"Loaded: {url}")
                            except Exception as e:
                                st.error(f"Failed to load {url}: {str(e)}")

                        if docs:
                            # Initialize components
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200
                            )
                            splits = text_splitter.split_documents(docs)

                            embeddings = OllamaEmbeddings(model=st.session_state.llm_model)
                            st.session_state.vector_db = Chroma.from_documents(
                                documents=splits,
                                embedding=embeddings,
                                persist_directory=st.session_state.vector_db_path
                            )

                            st.session_state.retriever = st.session_state.vector_db.as_retriever(
                                search_kwargs={"k": 5}
                            )
                            st.session_state.llm = OllamaLLM(model=st.session_state.llm_model)
                            st.session_state.prompt = ChatPromptTemplate.from_template(
                                """Answer the question based only on the following context:
                                {context}

                                Question: {question}

                                Provide a concise answer with relevant details."""
                            )
                            st.session_state.rag_initialized = True
                            st.success("RAG system initialized successfully!")

                    except Exception as e:
                        st.error(f"Initialization failed: {str(e)}")
                        st.session_state.rag_initialized = False

    # Query interface
    if st.session_state.rag_initialized:
        st.header("Ask Questions")
        query = st.text_input("Enter your question")

        if query:
            with st.spinner("Processing your query..."):
                try:
                    # Retrieve relevant docs
                    docs = st.session_state.retriever.invoke(query)

                    # Display results
                    st.subheader("Relevant Documents")
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Document {i}"):
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                            st.text(doc.page_content[:1000])

                    # Generate answer
                    context = "\n\n".join([d.page_content for d in docs])
                    answer = st.session_state.llm.invoke(
                        st.session_state.prompt.format(
                            context=context,
                            question=query
                        )
                    )
                    st.subheader("Answer")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    else:
        st.info("Please upload documents and initialize the RAG system in the sidebar")

if __name__ == "__main__":
    main()