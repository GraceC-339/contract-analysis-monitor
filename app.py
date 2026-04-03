import streamlit as st
import time
import os
# from dotenv import load_dotenv

# LangChain and Data tools
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Import Ollama instead of OpenAI for local LLM hosting
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# LCEL imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load OpenAI API Key from .env (Not using in this example since we're using Ollama, but keeping for reference if needed)
# load_dotenv()

st.set_page_config(layout="wide")
st.title("🛡️ PFI Contract Analysis & Monitor Platform")
st.write("Upload a contract to observe the pipeline steps and extract liability data.")

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Set up the two columns
col1, col2 = st.columns([1, 2])

# --- LEFT COLUMN: MONITORING THE PIPELINE ---
with col1:
    st.subheader("📊 Processing Pipeline Monitor")
    
    # We will use Streamlit file uploader to mimic real document intake
    uploaded_file = st.file_uploader("Upload Contract PDF", type=["pdf"])
    
    # Let's create visual placeholders for our fake "monitoring" metrics
    status_box = st.empty()
    metric_col1, metric_col2 = st.columns(2)
    m1 = metric_col1.empty()
    m2 = metric_col2.empty()
    
    # Initialize some session states so the UI remembers our data across clicks
    if "db" not in st.session_state:
        st.session_state.db = None # This will hold our vector database after processing
        st.session_state.chunks_count = 0 # Just to show how many chunks we created in the monitor
        st.session_state.active_filename = None

    if uploaded_file is not None:
        uploaded_filename = os.path.basename(uploaded_file.name)
        if uploaded_filename != st.session_state.active_filename:
            # New file selected: reset index so we rebuild for the active document.
            st.session_state.db = None
            st.session_state.active_filename = uploaded_filename

    if uploaded_file is not None and st.session_state.db is None:
        status_box.info("Pipeline triggered. Starting ingestion...")

        uploaded_filename = st.session_state.active_filename
        upload_path = os.path.join(UPLOADS_DIR, uploaded_filename)

        for filename in os.listdir(UPLOADS_DIR):
            existing_path = os.path.join(UPLOADS_DIR, filename)
            if (
                filename.lower().endswith(".pdf")
                and filename != uploaded_filename
                and os.path.isfile(existing_path)
            ):
                os.remove(existing_path)
        
        # Save uploaded file temporarily to read it
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        time.sleep(1) # Fake delay for visual effect
        
        # Step 1: Document Loading
        status_box.warning("🔄 Step 1: Parsing PDF structure...")
        loader = PyPDFLoader(upload_path)
        docs = loader.load()
        m1.metric("Pages Loaded", len(docs))
        time.sleep(1)
        
        # Step 2: Chunking
        status_box.warning("🔄 Step 2: Chunking contract into overlapping text blocks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        st.session_state.chunks_count = len(splits)
        m2.metric("Text Chunks Created", len(splits))
        time.sleep(1)
        
        # Step 3: Embeddings & Vector DB
        status_box.warning("🔄 Step 3: Generating mathematical embeddings & Indexing...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Create the FAISS vector database
        st.session_state.db = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        status_box.success("✅ Contract successfully indexed and ready for analysis!")
        
# --- RIGHT COLUMN: AI ANALYSIS ---
with col2:
    st.subheader("🤖 AI Contract Assisstant")

    if st.session_state.get("active_filename"):
        st.caption(f"Active document: {st.session_state.active_filename}")
    
    if st.session_state.db is not None:
        # Prompt user for a query
        user_query = st.text_input(
            "What would you like to know about this contract?", 
            "What are the liability limits or termination clauses in this document?"
        )
        
        if st.button("Run AI Analysis"):
            with st.spinner("Retrieving relevant contract chunks and generating answer..."):
                
                # 1. Setup the retriever from our database

                # Dynamically create a retriever that filters by the uploaded filename to ensure we only get relevant chunks for this query
                retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})
                
                #2. Define a helper function to format retrieved documents into one block of text
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # 3. Define the Brain (LLM)
                llm = OllamaLLM(model="llama3.2", temperature=0.1)
                
                # 4. Create prompt template
                system_prompt = (
                    "You are a contract analysis expert. Use the following context to answer the question. Explain the answer in simple terms."
                    "If the answer is not present in the context, state that it cannot be found. \n\n"
                    "Context: {context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # --- DEBUGGING ---
                print("=== DEBUG: RAG FLOW ===")
                # 1. Check if the database actually finds relevant chunks for the query
                test_docs = retriever.invoke(user_query)
                print(f"Retrieved {len(test_docs)} documents for query: '{user_query}'")

                for i, doc in enumerate(test_docs):
                    print(f"Doc {i+1} Metadata: {doc.metadata}")

                print("-------------------------------")

                # 5. The Modern LCEL Chain (Using the pipe operator for clarity)
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # 6. Run the chain with the user's query
                retrieved_docs = retriever.invoke(user_query)
                response_text = rag_chain.invoke(user_query)
                
                # Output the response
                st.markdown("### AI Analysis Result")
                st.write(response_text)
                
                # Show references (for auditing/data analysis purposes!)
                with st.expander("🔍 View Retrieved Contract Source Segments"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.write(doc.page_content)
                        st.divider()
    else:
        st.info("Please upload a PDF contract on the left to begin the analysis.")