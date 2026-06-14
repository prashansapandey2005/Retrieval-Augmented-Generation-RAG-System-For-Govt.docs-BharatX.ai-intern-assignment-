import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- CONFIGURATION ---
DB_PATH = "./qdrant_storage"     # Must match the folder created by your ingest script
COLLECTION_NAME = "gov_docs"

def setup_retriever():
    """Connects to the existing Qdrant database on your hard drive."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database folder '{DB_PATH}' not found. Run the ingestion script first!")

    print("Connecting to Qdrant Database...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # NEW: Using the updated QdrantVectorStore class to load existing DB
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    return vector_store

def ask_government_rag(query: str, vector_store, top_k: int = 15):
    """Retrieves documents, groups them, and prompts the LLM for a comparison."""
    
    # 1. Initialize LLM (Requires OPENAI_API_KEY environment variable)
    try:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, groq_api_key="UPLOAD_Key")
    except Exception as e:
        print("Error: Could not initialize LLM. Did you set your OPENAI_API_KEY?")
        return

    print(f"\nSearching database for: '{query}'...")
    
    # 2. Retrieve top chunks
    docs = vector_store.similarity_search(query, k=top_k)
    
    if not docs:
        print("No relevant documents found in the database.")
        return

    # 3. Group the contexts by source (as requested in the assignment)
    grouped_contexts = {}
    for doc in docs:
        src = doc.metadata.get("source_file", "Unknown Source")
        super_warn = doc.metadata.get("supersedes", None)
        
        if src not in grouped_contexts:
            grouped_contexts[src] = {"content": "", "supersedes": super_warn}
        
        grouped_contexts[src]["content"] += f"\n...{doc.page_content}..."

    # 4. Format the context string for the LLM
    context_str = ""
    for key, val in grouped_contexts.items():
        context_str += f"\n### Source Document: {key}\n"
        if val["supersedes"]:
            context_str += f"**METADATA WARNING: THIS DOCUMENT EXPLICITLY SUPERSEDES {val['supersedes']}**\n"
        context_str += f"Content:\n{val['content']}\n"

    print("Analyzing documents and generating response...\n")

    # 5. The strict Prompt Template for the LLM
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

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        # Connect to DB
        vector_db = setup_retriever()
        
        print("\n" + "="*50)
        print("GOVERNMENT RAG PIPELINE ACTIVE")
        print("="*50)
        
        # Test Query (Change this to test different things!)
        test_query = "What is the procedure for regulating tax refunds according to the 1980 rules vs the 2011 rules?"
        
        final_answer = ask_government_rag(test_query, vector_db)
        
        print("\n" + final_answer)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")