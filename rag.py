import streamlit as st
import random
import re
import requests
import json
from time import sleep

# --- RAG Dependency Note ---
# NOTE: To run this code, you must install the PDF reading library:
# pip install pypdf
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Please install the 'pypdf' library: pip install pypdf")
    PdfReader = None # Set to None if import fails


# --- Configuration (using placeholders for API key and URL) ---
API_KEY = "AIzaSyA0aVBVADHzgAN8-3hcf02VPh7sqnRa1FY" # Replace with your actual Gemini API Key
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
MAX_CHUNK_SIZE = 1500  # Max characters per document chunk

# --- Helper Functions ---

def text_splitter(text: str) -> list[str]:
    """Splits text into manageable chunks for RAG based on paragraph breaks."""
    
    chunks = []
    # Use a basic split by paragraph and then check size
    paragraphs = re.split(r'\n{2,}', text)
    
    current_chunk = ""
    for paragraph in paragraphs:
        if len(paragraph.strip()) == 0:
            continue
            
        # Check if adding the new paragraph exceeds the limit
        # We add 2 for the newline characters we insert later
        if len(current_chunk) + len(paragraph) + 2 < MAX_CHUNK_SIZE:
            current_chunk += paragraph + "\n\n"
        else:
            # If the current chunk is not empty, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start a new chunk with the current paragraph
            current_chunk = paragraph + "\n\n"
            
    # Save the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

@st.cache_data
def load_and_split_pdf(uploaded_file):
    """Loads a PDF, extracts text, and splits it into chunks."""
    if PdfReader is None:
        return []

    st.info(f"Loading and processing PDF: {uploaded_file.name}")
    try:
        pdf_reader = PdfReader(uploaded_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n\n"
        
        st.success(f"Extracted {len(full_text):,} characters from {len(pdf_reader.pages)} pages.")
        
        st.info(f"Splitting document into chunks (max {MAX_CHUNK_SIZE} chars)...")
        chunks = text_splitter(full_text)
        st.success(f"Document split into {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []


def retrieve_context(query: str, chunks: list[str], k: int = 3) -> list[str]:
    """
    SIMULATED RETRIEVAL:
    
    Finds the top 'k' chunks based on keyword matching.
    """
    query_words = set(query.lower().split())
    scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
        match_count = len(query_words.intersection(chunk_words))
        scores.append((match_count, i))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    
    top_indices = [index for score, index in scores if score > 0][:k]
    
    if not top_indices:
        st.warning("No keyword matches found. Retrieving a single random chunk for grounding.")
        return [random.choice(chunks)] if chunks else []
    
    context_chunks = [chunks[i] for i in top_indices]
    
    return context_chunks

def generate_rag_response(query: str, context: list[str]) -> str:
    """Constructs the prompt and calls the Gemini API."""
    
    if not context:
        st.error("Cannot generate response: No document context is available.")
        return ""

    # 1. CONSTRUCT THE RAG PROMPT
    context_text = "\n\n---\n\n".join(context)
    
    system_instruction = (
        "You are an expert Q&A assistant. "
        "Your goal is to answer the user's question by referring to the provided context. "
        "If the answer cannot be found in the context, clearly state, 'The provided documents do not contain enough information to answer this question.' "
        "Do not use external knowledge."
    )
    
    user_query_with_context = (
        f"Context:\n\n{context_text}\n\n"
        f"---"
        f"Question: {query}"
    )

    # 2. CONSTRUCT THE API PAYLOAD
    payload = {
        "contents": [{"parts": [{"text": user_query_with_context}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    
    # 3. CALL THE API (with retry logic)
    st.info("Sending request to Gemini API with retrieved context...")
    
    for attempt in range(3):
        try:
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status() 
            
            result = response.json()
            
            generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'API Response Error: Could not extract text.')
            return generated_text
            
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}. Status code: {response.status_code}. Response: {response.text}")
            break
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt + 1}: Request failed: {e}. Retrying...")
            sleep(2 ** attempt) 
            
    return "Failed to get a response from the Gemini API after multiple retries."


# --- Streamlit UI ---

st.set_page_config(
    page_title="Gemini PDF RAG Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Gemini PDF RAG Chatbot")
st.caption("Retrieval-Augmented Generation using uploaded PDF context and the Gemini API.")


# Initialize session state for document chunks
if 'doc_chunks' not in st.session_state:
    st.session_state.doc_chunks = []
if 'pdf_file' not in st.session_state:
    st.session_state.pdf_file = None


# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header("1. PDF Document Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload your PDF document:", 
        type="pdf"
    )

    if uploaded_file and uploaded_file != st.session_state.pdf_file:
        # File has been newly uploaded or changed
        st.session_state.pdf_file = uploaded_file
        
        with st.spinner("Z-Bot is processing..."):
            st.session_state.doc_chunks = load_and_split_pdf(uploaded_file)
            
    # Display the current status
    st.markdown("---")
    if st.session_state.pdf_file:
        st.metric("Loaded File", st.session_state.pdf_file.name)
    else:
        st.metric("Loaded File", "None")
        
    st.metric("Total Chunks Loaded", len(st.session_state.doc_chunks))


# --- Main Interface for Q&A ---
st.header("Question:")

if not st.session_state.doc_chunks:
    st.warning("Please upload and process a PDF document in the sidebar before asking a question.")
else:
    # Query Input
    query = st.text_input(
        "Enter your question about the PDF content:", 
        placeholder="E.g., What is the main finding of the report?"
    )
    
    if st.button("Generate Answer", type="primary"):
        if query:
            with st.spinner("Searching and generating response..."):
                # 1. Retrieval Step
                retrieved_context = retrieve_context(query, st.session_state.doc_chunks, k=3)
                
                # 2. Generation Step
                final_answer = generate_rag_response(query, retrieved_context)
            
            # Display Results
            st.subheader("🤖 Answer")
            st.success(final_answer)
            
            # Display Context Used
            with st.expander("🔍 Context Used for Generation"):
                if retrieved_context:
                    st.markdown(
                        f"**Retrieved {len(retrieved_context)} relevant chunks:**"
                    )
                    for i, chunk in enumerate(retrieved_context):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.code(chunk, language='text')
                else:
                    st.warning("No context was retrieved.")

# --- Footer and Disclaimer ---
st.markdown("---")
st.markdown("""
<style>
.stButton>button {
    background-color: #3B82F6; /* Adjusted button color for a change */
    color: white;
    border-radius: 8px;
    padding: 10px;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.caption("Architecture: PDF Upload -> Load/Extract Text -> Text Splitter -> (Simulated) Retrieval -> Gemini API Prompt -> Final Answer")
