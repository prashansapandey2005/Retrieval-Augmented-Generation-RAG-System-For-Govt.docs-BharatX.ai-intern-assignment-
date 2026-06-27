# Retrieval-Augmented-Generation-RAG-System-For-Govt.docs-
<div align="center">
  <h1>🏛️ Government Legal Document RAG Pipeline</h1>
  <p><i>A production-grade, supersession-aware Retrieval-Augmented Generation system.</i></p>
  <h3><a href="https://retrieval-augmented-generation-rag-system.streamlit.app/">🔴 Live App Link: Try it here!</a></h3>
</div>
<hr>

<h2>📋 Summary of the Work</h2>
<p>
  I have created and deployed a production level Retrieval-Augmented Generation (RAG) pipeline for the intricacies of government and legal documents. Unlike normal RAG implementations that blindly retrieve text, this system is “supersession-aware.” It tracks the life cycle of legal circulars properly, automatically sensing and alerting the user when an older policy has been superseded by a newer one.

<p>
  The pipeline features a dual-path ingestion engine that addresses the “dirty data” reality of government archives, processing both clean, digitally-born PDFs and legacy, scanned image documents. The end result is an interactive Streamlit UI that compares legal facts, provides exact document citations and formats output in clean Markdown.
</p>
</p>

<hr>

<h2>🏗️ Implementation and Approach</h2>
<p>
  My approach to architecture was to focus on modularity, data integrity, and hallucination reduction. The system is decoupled into two parts: an offline Ingestion Engine and a real-time Retrieval & Generation App.
</p>

<ul>
  <li>
    <b>1. Dual Path Ingestion & OCR fallback</b><br>
    <i>Approach:</i> Government datasets often contain scanned images that are not selectable. I did a dynamic layout check with PyPDFLoader In case the text payload of the first page of a document is less than 40 characters, the system will automatically send the file through an OCR fallback pipeline that uses <code>pdf2image</code> and <code>Tesseract OCR</code>.
    <i>Result:</i> Extraction success rate of 100% across both modern CBDT notifications and legacy scanned Corrigendums.
  </li>
  <br>
  <li>
    <b>2. Metadata Extraction & The Supersession Graph</b><br>
    <i>Approach:</i> Vector search is not good for matching exact serial numbers. Before chunking I extracted some structured metadata with regular expressions: <i>Document Type, Circular Number, Date</i>, <i>Superseded Targets</i> (using the legal phrasing “in supersession of”).<br>
    <i>Result:</i>  Each piece of text is linked to its exact legal identity. The system can then trace relationships between documents over decades.

  </li>
  <br>
  <li>
    <b>3. Semantic Chunking & Vector Storage</b><br>
    <i>Approach:</i> Text is split using <code>RecursiveCharacterTextSplitter</code> (3000 chars chunks and 400 chars overlap) to avoid breaking up complex legal clauses. These chunks are embedded using <code>HuggingFace's sentence-transformers/all-mpnet-base-v2</code> (selected for its superior performance on formal/legal semantics) and stored persistently in a local <strong>Qdrant Vector Database</strong>.
  </li>
  <br>
  <li>
    <b>4. Retrieval & Context Grouping</b><br>
    <i>Approach:</i> When asked, Qdrant finds the top 15 relevant chunks. Crucially, the retrieval engine clusters these chunks by their source file before passing them to the LLM. This prevents the LLM from mixing up the 1980 rules and the 2011 rules.
  </li>
  <br>
  <li>
    <b>5. Augmentation & Generation (LLM)</b><br>
    <i>Approach:</i> I used Groq API (Llama 3.3 70B) for super fast inference. The system prompt compels the LLM to act like a strict legal analyst. In the case where the injected context contains a supersession metadata tag, the LLM is instructed to halt its normal comparison, and insert a prominent  <b>SUPERSESSION WARNING</b>  at the top of the UI, explaining exactly which rule is dead and which is ative.
  </li>
  <br>
  <li>
    <b>6. User Interface</b><br>
    <i>Approach:</i> Built a lightweight interactive frontend using <strong>Streamlit</strong> to enable non-technical stakeholders to query the database, securely enter API keys and view Markdown-rendered comparison tables and alerts in real-time.
  </li>
</ul>

<hr>

<h2>📂 Repository Structure</h2>

<pre><code>gov_rag_project/
│
├── Dataset_govt/              # Directory for raw government PDFs
├── qdrant_storage/              # Persistent local vector database (Generated)
├── app.py                       # Streamlit UI & Retrieval Logic
├── Ingest.py                    # Dual-path Ingestion & Embedding Logic
|── retreive.py                  # user Query and generation logic(terminal)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
</code></pre>

<hr>

<h2>🚀 How to Run Locally</h2>

<h3>1. Prerequisites</h3>
<p>
  Ensure you have Python 3.8+ installed. For the OCR fallback to work on scanned PDFs, you must have the following system dependencies installed and added to your system PATH:
</p>
<ul>
  <li><b>Tesseract OCR</b></li>
  <li><b>Poppler</b> (for <code>pdf2image</code>)</li>
</ul>

<h3>2. Setup the Environment</h3>
<p>Clone the repository and set up a virtual environment:</p>

<pre><code># Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Mac/Linux)
source venv/bin/activate
</code></pre>

<h3>3. Install Dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<h3>4. Build the Database (Ingestion)</h3>
<p>Place your target government PDFs into the <code>my_rag_dataset/</code> folder, then run the ingestion script to build the Qdrant vector database:</p>
<pre><code>python ingest.py
</code></pre>

<h3>5. Run the Application</h3>
<p>Launch the Streamlit interface:</p>
<pre><code>streamlit run app.py
</code></pre>
<p><i>Note: You will need a valid Groq API key to enter into the UI sidebar to generate the LLM responses.</i></p>

<hr>

<h2> App Output / Screenshots</h2>
<p>Below are screenshots demonstrating the live application in action, including the interactive UI, the generated comparison tables, and the supersession warnings.</p>

<br>
<img width="1366" height="768" alt="Screenshot (449)" src="https://github.com/user-attachments/assets/3e53ced4-4df0-4c18-8e31-d0d47b0042a7" />
<br>
<img width="1366" height="768" alt="Screenshot (448)" src="https://github.com/user-attachments/assets/6358fd15-7897-47c2-a110-57e30a82b936" />
<br>
<img width="1366" height="768" alt="Screenshot (447)" src="https://github.com/user-attachments/assets/371b1270-212f-49cd-958b-81da6703f37a" />
<br>
<img width="1366" height="768" alt="Screenshot (446)" src="https://github.com/user-attachments/assets/da54bae3-af81-4d73-b9d1-b60eb01c4d3f" />
