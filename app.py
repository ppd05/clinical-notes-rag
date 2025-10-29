#Imports
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json
import os
from typing import List, Dict
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

# Loading the environment variables specifically the API Key (free tier)
load_dotenv()

# Page configuration for streamlit
st.set_page_config(page_title="Clinical Notes RAG System", layout="wide")

# Initialize session state
if 'structured_notes' not in st.session_state:
    st.session_state.structured_notes = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'last_process_time' not in st.session_state:
    st.session_state.last_process_time = None
if 'cooldown_end_time' not in st.session_state:
    st.session_state.cooldown_end_time = None

# Rate limiting configuration for the API
MAX_PATIENTS_PER_SESSION = 2
COOLDOWN_MINUTES = 15  # Wait time after hitting limit per user to avoid overusage.

def get_api_key():
    """Get API key from environment or Streamlit secrets"""
    # Try to get from environment variable first (for local)
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    
    # Try Streamlit secrets (for deployment)
    try:
        return st.secrets.get("GEMINI_API_KEY", None)
    except:
        return None

def test_api_key(api_key: str) -> bool:#API Key testing validity
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content("Hello")
        return True
    except Exception as e:
        st.error(f"API Key Test Failed: {str(e)}")
        return False

def check_rate_limit(): #User rate limit code
    current_time = datetime.now()
    
    # Check if in cooldown period
    if st.session_state.cooldown_end_time:
        if current_time < st.session_state.cooldown_end_time:
            remaining = (st.session_state.cooldown_end_time - current_time).total_seconds() / 60
            return False, f"⏳ Please wait {int(remaining)} minute(s) before processing more patients."
        else:
            # Cooldown expired, reset
            st.session_state.cooldown_end_time = None
            st.session_state.usage_count = 0
    
    # Check usage count
    if st.session_state.usage_count >= MAX_PATIENTS_PER_SESSION:
        # Set cooldown
        st.session_state.cooldown_end_time = current_time + timedelta(minutes=COOLDOWN_MINUTES)
        return False, f"⏳ Usage limit reached! You can process {MAX_PATIENTS_PER_SESSION} patients per session. Please wait {COOLDOWN_MINUTES} minutes."
    
    return True, "OK"

def get_embeddings(): #I am using a HuggingFaceEmbedding model that is free of cost than the usual big runners.
    try:
        with st.spinner("Loading embedding model (first time may take a minute)..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def extract_structured_data(text: str, api_key: str) -> Dict: #Unstructured to structured notes
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        prompt = f"""
        Extract structured information from the following clinical note and return it in JSON format.
        Include: patient_id, date, chief_complaint, symptoms, diagnosis, medications, and treatment_plan.
        If any field is not present, use "Not specified".
        
        Clinical Note:
        {text}
        
        Return only valid JSON without any markdown formatting.
        """
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                structured_data = json.loads(response_text)
                return structured_data
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    st.warning(f"Could not extract structured data: {str(e)}")
                    return {
                        "patient_id": "Unknown",
                        "raw_text": text[:200] + "...",
                        "error": f"Could not parse structured data: {str(e)}"
                    }
    except Exception as e:
        st.error(f"Error in extract_structured_data: {str(e)}")
        return {
            "patient_id": "Unknown",
            "error": str(e)
        }

def process_clinical_notes(uploaded_files, api_key: str, embeddings): #Process the uploaded files
    if not embeddings:
        st.error("Embeddings not configured properly")
        return None, {}
    
    all_documents = []
    structured_notes = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Read the text file
            text_content = uploaded_file.read().decode('utf-8')
            
            # Extract structured data with error handling
            try:
                structured_data = extract_structured_data(text_content, api_key)
                patient_id = structured_data.get('patient_id', f'patient_{idx}')
                structured_notes[patient_id] = structured_data
            except Exception as e:
                st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                patient_id = f'patient_{idx}'
                structured_notes[patient_id] = {
                    "patient_id": patient_id,
                    "error": str(e),
                    "raw_text": text_content[:200]
                }
            
            # Create document with metadata
            doc = Document(
                page_content=text_content,
                metadata={
                    'patient_id': patient_id,
                    'source': uploaded_file.name,
                    'structured_data': json.dumps(structured_notes[patient_id])
                }
            )
            
            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents([doc])
            all_documents.extend(splits)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            time.sleep(0.5)  # Avoid rate limiting
        
        status_text.text("Creating vector store...")
        
        # Creating FAISS vector store
        vector_store = FAISS.from_documents(all_documents, embeddings)
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        return vector_store, structured_notes
        
    except Exception as e:
        st.error(f"Error processing clinical notes: {str(e)}")
        return None, {}

def summarize_notes(patient_id: str, notes: Dict, api_key: str) -> str: #Summarize the uploaded clinical notes for the user.
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite') #Using free tier model since pro and rest are paid.
        
        prompt = f"""
        Provide a comprehensive medical summary of the following structured clinical note:
        
        {json.dumps(notes, indent=2)}
        
        Include:
        1. Patient Overview
        2. Key Medical Findings
        3. Prescribed Treatment
        4. Recommendations
        
        Keep it professional and concise.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def query_rag(question: str, patient_id: str, vector_store, api_key: str) -> str: #Query the RAG
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Retriever to retrieve relevant documents
        if patient_id and patient_id != "All Patients":
            docs = vector_store.similarity_search(question, k=3)
            # Option to Filter by patient_id
            docs = [doc for doc in docs if doc.metadata.get('patient_id') == patient_id]
        else:
            docs = vector_store.similarity_search(question, k=3)
        
        if not docs:
            return "No relevant information found in the clinical notes."
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Based on the following clinical notes, answer the question accurately and professionally.
        
        Clinical Notes Context:
        {context}
        
        Question: {question}
        
        Provide a clear, medically accurate answer based only on the information provided.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error querying RAG system: {str(e)}"

# Streamlit UI
st.title("🏥 Clinical Notes RAG System")
st.markdown("Upload clinical notes, extract structured data, and query with AI")
st.info("🚀 Using Gemini 2.0 Flash-Lite + Free HuggingFace Embeddings")

# Get API key from environment or secrets
env_api_key = get_api_key()

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if env_api_key:
        st.success("✅ API Key loaded from environment")
        st.session_state.api_key = env_api_key
    else:
        api_key_input = st.text_input("Enter Gemini API Key", type="password")
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key configured!")
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. API key will be loaded automatically
    2. Upload clinical notes (.txt files)
    3. Process the notes
    4. View structured data & summaries
    5. Ask questions about the notes
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ Usage Limits")
    st.markdown(f"""
    - **Max Files per Session:** {MAX_PATIENTS_PER_SESSION} patients
    - **Cooldown Period:** {COOLDOWN_MINUTES} minutes
    - **Current Usage:** {st.session_state.usage_count}/{MAX_PATIENTS_PER_SESSION}
    """)
    
    if st.session_state.cooldown_end_time:
        remaining_time = (st.session_state.cooldown_end_time - datetime.now()).total_seconds() / 60
        if remaining_time > 0:
            st.warning(f"⏳ Cooldown: {int(remaining_time)} min remaining")
    
    st.markdown("---")
    st.markdown("**Model:** Gemini 2.0 Flash-Lite")
    st.markdown("**Embeddings:** HuggingFace (Free)")

# Main content
if not st.session_state.api_key:
    st.warning("⚠️ Please configure your Gemini API key. Add it to .env file locally or Streamlit secrets for deployment.")
    st.stop()

# Configure Gemini
genai.configure(api_key=st.session_state.api_key)

# Load embeddings (only once)
if not st.session_state.embeddings:
    st.session_state.embeddings = get_embeddings()

embeddings = st.session_state.embeddings

if not embeddings:
    st.error("Failed to load embeddings. Please refresh the page.")
    st.stop()

# File upload section
st.header("📁 Upload Clinical Notes")
uploaded_files = st.file_uploader(
    "Upload clinical note files (.txt)",
    type=['txt'],
    accept_multiple_files=True
)

if uploaded_files:
    # Check rate limit before processing
    can_process, message = check_rate_limit()
    
    # Display usage status
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Files Uploaded", len(uploaded_files))
    with col_info2:
        remaining = MAX_PATIENTS_PER_SESSION - st.session_state.usage_count
        st.metric("Remaining Processes", max(0, remaining))
    
    if not can_process:
        st.warning(message)
        st.info("💡 **Tip:** Refresh the page after cooldown period to reset your usage.")
    else:
        if st.button("🔄 Process Clinical Notes", type="primary"):
            # Check limit again at button click
            can_process_now, message_now = check_rate_limit()
            if not can_process_now:
                st.error(message_now)
            else:
                # Check if number of files exceeds remaining quota
                files_to_process = len(uploaded_files)
                remaining_quota = MAX_PATIENTS_PER_SESSION - st.session_state.usage_count
                
                if files_to_process > remaining_quota:
                    st.error(f"❌ You can only process {remaining_quota} more file(s) in this session. Please select fewer files or wait {COOLDOWN_MINUTES} minutes.")
                else:
                    with st.spinner("Processing clinical notes..."):
                        vector_store, structured_notes = process_clinical_notes(
                            uploaded_files,
                            st.session_state.api_key,
                            embeddings
                        )
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.structured_notes.update(structured_notes)
                            
                            # Increment usage count
                            st.session_state.usage_count += files_to_process
                            st.session_state.last_process_time = datetime.now()
                            
                            st.success(f"✅ Processed {files_to_process} file(s) successfully!")
                            st.info(f"ℹ️ You have {MAX_PATIENTS_PER_SESSION - st.session_state.usage_count} process(es) remaining in this session.")
                        else:
                            st.error("Failed to process clinical notes. Please check the error messages above.")

# Display structured data and summaries
if st.session_state.structured_notes:
    st.header("📊 Structured Clinical Data")
    
    # Patient selector
    patient_ids = list(st.session_state.structured_notes.keys())
    selected_patient = st.selectbox("Select Patient", patient_ids)
    
    if selected_patient:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Structured Data")
            st.json(st.session_state.structured_notes[selected_patient])
        
        with col2:
            st.subheader("Summary")
            if st.button("📝 Generate Summary", key=f"summary_{selected_patient}"):
                with st.spinner("Generating summary..."):
                    summary = summarize_notes(
                        selected_patient,
                        st.session_state.structured_notes[selected_patient],
                        st.session_state.api_key
                    )
                    st.markdown(summary)

# Question answering section
if st.session_state.vector_store:
    st.header("💬 Ask Questions About Clinical Notes")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Enter your question")
    with col2:
        query_patient = st.selectbox(
            "Filter by Patient",
            ["All Patients"] + list(st.session_state.structured_notes.keys()),
            key="query_patient"
        )
    
    if st.button("🔍 Search", type="primary"):
        if question:
            with st.spinner("Searching clinical notes..."):
                answer = query_rag(
                    question,
                    query_patient if query_patient != "All Patients" else None,
                    st.session_state.vector_store,
                    st.session_state.api_key
                )
                st.markdown("### Answer:")
                st.markdown(answer)
        else:
            st.warning("Please enter a question!")

# Footer
st.markdown("---")
st.markdown("*Powered by Gemini 2.0 Flash-Lite, HuggingFace Embeddings, LangChain, and FAISS*")