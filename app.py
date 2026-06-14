import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Govt Doc RAG", page_icon="🏛️", layout="wide")

# --- CACHE THE DATABASE LOADING ---
# This ensures the database only loads once, making the app super fast!
@st.cache_resource
def load_database():
    db_path = "./qdrant_storage"
    if not os.path.exists(db_path):
        st.error(f"Database folder '{db_path}' not found. Please run ingest.py first.")
        st.stop()
        
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=db_path,
        collection_name="gov_docs"
    )
    return vector_store

# --- RAG LOGIC ---
def ask_government_rag(query, vector_store, groq_key):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0, 
        groq_api_key=groq_key
    )
    
    # Retrieve top chunks
    docs = vector_store.similarity_search(query, k=15)
    if not docs:
        return "No relevant documents found in the database."

    # Group the contexts by source
    grouped_contexts = {}
    for doc in docs:
        src = doc.metadata.get("source_file", "Unknown Source")
        super_warn = doc.metadata.get("supersedes", None)
        
        if src not in grouped_contexts:
            grouped_contexts[src] = {"content": "", "supersedes": super_warn}
        grouped_contexts[src]["content"] += f"\n...{doc.page_content}..."

    # Format the context string
    context_str = ""
    for key, val in grouped_contexts.items():
        context_str += f"\n### Source Document: {key}\n"
        if val["supersedes"]:
            context_str += f"**METADATA WARNING: THIS DOCUMENT EXPLICITLY SUPERSEDES {val['supersedes']}**\n"
        context_str += f"Content:\n{val['content']}\n"

    # Prompt Template
    prompt_template = PromptTemplate.from_template("""
    You are an expert Indian Government Legal and Bureaucratic Analyst. 
    Answer the user's query strictly based on the extracted documents provided below.
    
    Context Documents:
    {context}
    
    User Query: {query}
    
    Strict Instructions:
    1. Compare the facts, policies, and events between the sources provided.
    2. Explicitly highlight differences using markdown (e.g., > Blockquotes or **Bold** text for diffs).
    3. If the context contains a "METADATA WARNING" that a newer document supersedes an older one, you MUST provide a prominent "🚨 SUPERSESSION WARNING 🚨" at the top of your response explaining which rule is dead and which is active.
    4. Include precise citations (Document names, Circular Numbers, and Dates) for all claims.
    5. Output the final comparison in a clean, readable Markdown format.
    """)
    
    chain = prompt_template | llm
    response = chain.invoke({"context": context_str, "query": query})
    
    return response.content

# --- UI LAYOUT ---
st.title("🏛️ Government Document RAG Pipeline")
st.markdown("""
This system intelligently retrieves government policies, compares facts, and automatically triggers warnings if a policy has been **superseded** by a newer circular.
""")

# Sidebar for Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("- Qdrant Vector DB")
    st.markdown("- HuggingFace Embeddings")
    st.markdown("- Groq Llama 3.3")

# Main Interface
vector_db = load_database()

user_query = st.text_input("Enter your legal or policy query:", placeholder="e.g., What is the procedure for regulating tax refunds according to the 1980 rules vs the 2011 rules?")

if st.button("Analyze Documents"):
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar first.")
    elif not user_query:
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("Searching database and analyzing legal text..."):
            try:
                answer = ask_government_rag(user_query, vector_db, api_key)
                st.success("Analysis Complete!")
                st.markdown("---")
                # This is where the magic happens: Streamlit renders the markdown perfectly
                st.markdown(answer) 
            except Exception as e:
                st.error(f"An error occurred: {e}")