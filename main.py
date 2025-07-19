import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import fitz  # PyMuPDF
import sqlite3
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import hashlib
from typing import List, Dict, Any
import re

load_dotenv()  # Load all environment variables

# Set custom page config with emoji favicon
st.set_page_config(
    page_title="AI Content Summarizer ",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .content-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
    }
    
    /* Chat styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
    }
    
    .bot-message {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Error message styling */
    .error-box {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Summary result styling */
    .summary-result {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Statistics styling */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* History item styling */
    .history-item {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# RAG System Classes
class AdaptiveRAG:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_store = None
        self.documents = []
        self.document_metadata = []
        self.chunk_size = 1000
        self.overlap_size = 200
        
    def create_chunks(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'source': source_info['source'],
                'type': source_info['type'],
                'chunk_id': len(chunks),
                'word_count': len(chunk_words)
            })
        
        return chunks
    
    def add_document(self, text: str, source_info: Dict[str, Any]):
        """Add a document to the RAG system"""
        chunks = self.create_chunks(text, source_info)
        
        for chunk in chunks:
            self.documents.append(chunk['text'])
            self.document_metadata.append({
                'source': chunk['source'],
                'type': chunk['type'],
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count']
            })
        
        self._build_vector_store()
    
    def _build_vector_store(self):
        """Build FAISS vector store"""
        if not self.documents:
            return
        
        embeddings = self.embedding_model.encode(self.documents)
        dimension = embeddings.shape[1]
        
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings.astype('float32'))
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a query"""
        if not self.vector_store or not self.documents:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.vector_store.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(score),
                    'metadata': self.document_metadata[idx]
                })
        
        return results
    
    def generate_response(self, query: str, api_key: str) -> str:
        """Generate response using retrieved context, fallback to web search if needed"""
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=3)
        if not relevant_chunks:
            # Fallback to web search
            web_answer = web_search_answer(query, api_key)
            return f"🌐 _No answer found in your uploaded documents. This answer is from web search:_\n\n{web_answer}"
        
        # Build context from retrieved chunks
        context = ""
        sources = set()
        for chunk in relevant_chunks:
            context += f"Content from {chunk['metadata']['source']} ({chunk['metadata']['type']}):\n"
            context += f"{chunk['text']}\n\n"
            sources.add(f"{chunk['metadata']['source']} ({chunk['metadata']['type']})")
        
        # Create prompt
        prompt = f"""Based on the following context from uploaded documents, please answer the user's question accurately and comprehensively.

Context:
{context}

User Question: {query}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, mention what information might be missing. Always cite which sources your answer comes from.

Answer:"""
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            # Add source information
            source_list = "\n\n📚 **Sources used:**\n" + "\n".join([f"• {source}" for source in sources])
            
            return response.text + source_list
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clear_documents(self):
        """Clear all documents from the RAG system"""
        self.documents = []
        self.document_metadata = []
        self.vector_store = None

# Database setup
def init_database():
    conn = sqlite3.connect('summarizer.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_type TEXT NOT NULL,
            source_url TEXT,
            file_name TEXT,
            summary TEXT NOT NULL,
            word_count INTEGER,
            char_count INTEGER,
            compression_ratio REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add table for RAG documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            content_type TEXT NOT NULL,
            source_url TEXT,
            file_name TEXT,
            content_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_summary(content_type, source_url, file_name, summary, word_count, char_count, compression_ratio):
    conn = sqlite3.connect('summarizer.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO summaries (content_type, source_url, file_name, summary, word_count, char_count, compression_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (content_type, source_url, file_name, summary, word_count, char_count, compression_ratio))
    conn.commit()
    conn.close()

def save_rag_document(content_type, source_url, file_name, content_text):
    """Save document content for RAG system"""
    content_hash = hashlib.md5(content_text.encode()).hexdigest()
    
    conn = sqlite3.connect('summarizer.db')
    cursor = conn.cursor()
    
    # Check if document already exists
    cursor.execute('SELECT id FROM rag_documents WHERE content_hash = ?', (content_hash,))
    if cursor.fetchone():
        conn.close()
        return False  # Document already exists
    
    cursor.execute('''
        INSERT INTO rag_documents (content_hash, content_type, source_url, file_name, content_text)
        VALUES (?, ?, ?, ?, ?)
    ''', (content_hash, content_type, source_url, file_name, content_text))
    
    conn.commit()
    conn.close()
    return True  # Document saved successfully

def get_rag_documents():
    """Get all RAG documents"""
    conn = sqlite3.connect('summarizer.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT content_type, source_url, file_name, content_text, created_at
        FROM rag_documents ORDER BY created_at DESC
    ''')
    documents = cursor.fetchall()
    conn.close()
    return documents

def get_summary_history():
    conn = sqlite3.connect('summarizer.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM summaries ORDER BY created_at DESC LIMIT 10
    ''')
    history = cursor.fetchall()
    conn.close()
    return history

# Initialize database
init_database()

# Initialize RAG system in session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = AdaptiveRAG()
    # Load existing documents into RAG system
    documents = get_rag_documents()
    for doc in documents:
        content_type, source_url, file_name, content_text, created_at = doc
        source_info = {
            'source': source_url or file_name or 'Unknown',
            'type': content_type
        }
        st.session_state.rag_system.add_document(content_text, source_info)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Improved video ID extraction
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    return None

# Extract text from website/URL
def extract_website_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from body
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if not text or len(text) < 50:
            return None, "Could not extract meaningful text from this website."
        
        return text, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Error accessing website: {str(e)}"
    except Exception as e:
        return None, f"Error extracting text from website: {str(e)}"

# Extract text from PDF
def extract_pdf_text(pdf_file):
    try:
        # Read the uploaded file
        pdf_content = pdf_file.read()
        
        # Try PyMuPDF first (better text extraction)
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if text.strip():
                return text, None
        except:
            pass
        
        # Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            if text.strip():
                return text, None
            else:
                return None, "Could not extract text from PDF. The PDF might be scanned or image-based."
                
        except Exception as e:
            return None, f"Error reading PDF: {str(e)}"
            
    except Exception as e:
        return None, f"Error processing PDF file: {str(e)}"

# Fetch transcript
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            st.error("Invalid YouTube URL provided.")
            return None

        try:
            # First, try to get the transcript in English
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            st.info("Using English transcript.")
        except NoTranscriptFound:
            st.warning("English transcript not available. Trying to find another language.")
            # If English is not found, get the list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Find the language code of the first available transcript
            first_available_language = None
            for transcript in transcript_list:
                first_available_language = transcript.language_code
                st.info(f"Using transcript in '{transcript.language} ({transcript.language_code})'")
                break

            if not first_available_language:
                st.error("No transcripts could be found for this video.")
                return None
            
            # Fetch the transcript for that language
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[first_available_language])

        transcript_text = " ".join([item["text"] for item in transcript_data])
        return transcript_text
        
    except (TranscriptsDisabled, VideoUnavailable) as e:
        st.error(str(e))
        return None
    except NoTranscriptFound:
        st.error("No transcripts could be found for this video.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Generate summary using Gemini
def generate_gemini_content(text, prompt):
    user_key = st.session_state.get("user_gemini_api_key")
    if not user_key:
        st.error("Gemini API key not found. Please enter your key in the sidebar.")
        return None
    genai.configure(api_key=user_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt + text)
        return response.text
    except ResourceExhausted:
        st.error(
            "API quota exceeded. You have made too many requests. "
            "Please wait a while or check your billing details at https://ai.google.dev/gemini-api/docs/rate-limits"
        )
        return None
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {e}")
        return None

# --- Web search fallback using Gemini API ---
def web_search_answer(query, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Use Gemini's web search tool by instructing the model
        prompt = f"""
You are an AI assistant with access to the web. Please answer the following question using up-to-date information from the web. Be concise and only provide the required information. If you use any sources, mention them if possible.

Question: {query}

Answer concisely:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error using web search: {str(e)}"

def get_prompt(content_type, specific_type=""):
    if content_type == "YouTube Video":
        return '''You are a YouTube video content analyzer. Based on the provided transcript text, determine the video type (entertaining, musical, or educational/informational) and perform the following: 
        - Entertaining videos: Write a lively, engaging summary in paragraph form highlighting key events, funny moments, and the overall narrative.
        - Musical videos: Extract and format the song lyrics in paragraph form, excluding non-lyrical content like intros, outros, or spoken parts.
        - Educational/informational videos: Provide a clear, structured summary in paragraph form and bullet points of all theoretical concepts and main points discussed.
        Analyze the transcript and deliver the appropriate output for the video type.'''
    elif content_type == "Website/URL":
        return """You are a website content summarizer. Based on the provided webpage text, create a comprehensive summary focusing on the main topics, key information, and important details. Present the summary in paragraph form and bullet points for clarity."""
    elif content_type == "PDF Document":
        return """You are a PDF document summarizer. Based on the provided PDF text, create a detailed summary focusing on the main topics, key concepts, important findings, and conclusions. Present the summary in paragraph form and bullet points for clarity."""
    else:
        return """You are a content summarizer. Based on the provided text, create a comprehensive summary focusing on the main topics, key information, and important details. Present the summary in paragraph form and bullet points for clarity."""

# --- Sidebar: API Key Entry and RAG Status ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🔑 Gemini API Key</h2>
        <p>Enter your Google Gemini API key to use the app.</p>
    </div>
    """, unsafe_allow_html=True)
    user_api_key = st.text_input(
        "Your Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here...",
        key="user_gemini_api_key_input"
    )
    if user_api_key:
        st.session_state["user_gemini_api_key"] = user_api_key
    
    # RAG System Status
    if st.session_state.get("user_gemini_api_key"):
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🤖 RAG Chatbot</h2>
            <p>Documents loaded for Q&A</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show loaded documents
        rag_docs = get_rag_documents()
        st.write(f"**📚 Loaded Documents: {len(rag_docs)}**")
        
        if rag_docs:
            for i, (content_type, source_url, file_name, _, created_at) in enumerate(rag_docs[-5:]):  # Show last 5
                source_display = source_url or file_name or "Unknown"
                if len(source_display) > 30:
                    source_display = source_display[:27] + "..."
                st.write(f"• {content_type}: {source_display}")
        
        if st.button("🗑️ Clear All Documents"):
            conn = sqlite3.connect('summarizer.db')
            cursor = conn.cursor()
            cursor.execute('DELETE FROM rag_documents')
            conn.commit()
            conn.close()
            st.session_state.rag_system.clear_documents()
            st.rerun()
        
        # Show summary history
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>📚 Summary History</h2>
            <p>Your recent summaries</p>
        </div>
        """, unsafe_allow_html=True)
        history = get_summary_history()
        if history:
            for item in history:
                with st.expander(f"📄 {item[1]} - {item[8][:10]}"):
                    st.write(f"**Source:** {item[2] or item[3] or 'N/A'}")
                    st.write(f"**Words:** {item[5]:,}")
                    st.write(f"**Compression:** {item[7]:.1f}%")
                    st.write("**Summary:**")
                    st.text_area("", value=item[4], height=100, key=f"history_{item[0]}")
        else:
            st.info("No summaries yet. Create your first one!")
    else:
        st.info("Enter your Gemini API key above to unlock the app.")

# --- Block app if no API key ---
if not st.session_state.get("user_gemini_api_key"):
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    ">
        <h3 style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🔒 API Key Required</h3>
        <p style="font-size: 1.2rem; opacity: 0.9; margin: 0;">Please enter your Gemini API key in the sidebar to use the summarizer.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Main content
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Content Summarizer </h1>
    <p>Transform YouTube videos, websites, and PDFs into concise, intelligent summaries with RAG-powered Q&A</p>
</div>
""", unsafe_allow_html=True)

# Tab selection
tab1, tab2 = st.tabs(["📄 Content Summarizer", "🤖 RAG Chatbot"])

with tab1:
    # Initialize session state for content selection
    if 'selected_content' not in st.session_state:
        st.session_state.selected_content = None

    # Content type selection with start buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="content-card">
            <h3>🎥 YouTube Videos</h3>
            <p>Extract transcripts and create engaging summaries</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start YouTube Summarizer", key="youtube_start", use_container_width=True):
            st.session_state.selected_content = "youtube"

    with col2:
        st.markdown("""
        <div class="content-card">
            <h3>🌐 Websites</h3>
            <p>Scrape and summarize web content instantly</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Website Summarizer", key="website_start", use_container_width=True):
            st.session_state.selected_content = "website"

    with col3:
        st.markdown("""
        <div class="content-card">
            <h3>📄 PDF Documents</h3>
            <p>Process and summarize PDF files with ease</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start PDF Summarizer", key="pdf_start", use_container_width=True):
            st.session_state.selected_content = "pdf"

    # Show content based on selection
    if st.session_state.selected_content == "youtube":
        content_type = "🎥 YouTube Video"
    elif st.session_state.selected_content == "website":
        content_type = "🌐 Website/URL"
    elif st.session_state.selected_content == "pdf":
        content_type = "📄 PDF Document"
    else:
        content_type = None

    # Initialize variables
    text_content = None
    error_message = None
    source_url = None
    file_name = None

    # Show back button if content is selected
    if st.session_state.selected_content:
        if st.button("← Back to Selection", key="back_btn"):
            st.session_state.selected_content = None
            st.rerun()

    if content_type == "🎥 YouTube Video":
        st.markdown("""
        <div class="content-card">
            <h2>🎥 YouTube Video Summarizer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        youtube_link = st.text_input("🔗 **Enter YouTube Video Link:**", placeholder="Paste your YouTube video URL here...")
          
        add_to_rag = st.checkbox("🤖 Add to RAG Chatbot for Q&A", value=True)
        
        if youtube_link:
            video_id = extract_video_id(youtube_link)
            if video_id:
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True, caption="Video Preview")
            else:
                st.error("❌ Invalid YouTube link.")
        
        if st.button("✨ Get YouTube Summary"):
            if youtube_link:
                source_url = youtube_link
                text_content = extract_transcript_details(youtube_link)
                if text_content:
                    # Use a general prompt for all video types
                    prompt = get_prompt("YouTube Video", "Any")
                    # Add to RAG system if requested
                    if add_to_rag and text_content:
                        source_info = {
                            'source': youtube_link,
                            'type': 'YouTube Video'
                        }
                        if save_rag_document('YouTube Video', youtube_link, None, text_content):
                            st.session_state.rag_system.add_document(text_content, source_info)
                            st.success("✅ Video content added to RAG chatbot!")
                        else:
                            st.info("ℹ️ This video is already in the RAG system.")
            else:
                st.error("Please enter a YouTube URL first.")

    elif content_type == "🌐 Website/URL":
        st.markdown("""
        <div class="content-card">
            <h2>🌐 Website/URL Summarizer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        website_url = st.text_input("🔗 **Enter Website URL:**", placeholder="https://example.com")
        add_to_rag = st.checkbox("🤖 Add to RAG Chatbot for Q&A", value=True)
        
        if st.button("✨ Get Website Summary"):
            if website_url:
                source_url = website_url
                with st.spinner("⏳ Extracting content from website..."):
                    text_content, error_message = extract_website_text(website_url)
                if error_message:
                    st.error(f"❌ {error_message}")
                elif add_to_rag and text_content:
                    source_info = {
                        'source': website_url,
                        'type': 'Website/URL'
                    }
                    if save_rag_document('Website/URL', website_url, None, text_content):
                        st.session_state.rag_system.add_document(text_content, source_info)
                        st.success("✅ Website content added to RAG chatbot!")
                    else:
                        st.info("ℹ️ This website is already in the RAG system.")
            else:
                st.error("Please enter a website URL first.")

    elif content_type == "📄 PDF Document":
        st.markdown("""
        <div class="content-card">
            <h2>📄 PDF Document Summarizer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("📁 **Upload PDF Document:**", type=['pdf'])
        add_to_rag = st.checkbox("🤖 Add to RAG Chatbot for Q&A", value=True)
        
        if st.button("✨ Get PDF Summary"):
            if pdf_file:
                file_name = pdf_file.name
                with st.spinner("⏳ Extracting content from PDF..."):
                    text_content, error_message = extract_pdf_text(pdf_file)
                if error_message:
                    st.error(f"❌ {error_message}")
                elif add_to_rag and text_content:
                    source_info = {
                        'source': file_name,
                        'type': 'PDF Document'
                    }
                    if save_rag_document('PDF Document', None, file_name, text_content):
                        st.session_state.rag_system.add_document(text_content, source_info)
                        st.success("✅ PDF content added to RAG chatbot!")
                    else:
                        st.info("ℹ️ This PDF is already in the RAG system.")
            else:
                st.error("Please upload a PDF file first.")

    # Generate summary if we have content
    if text_content and not error_message:
        with st.spinner("⏳ Generating summary, please wait..."):
            # Determine the content type for prompt
            if content_type == "🎥 YouTube Video":
                prompt = get_prompt("YouTube Video", "Any")
                content_type_clean = "YouTube Video"
            elif content_type == "🌐 Website/URL":
                prompt = get_prompt("Website/URL")
                content_type_clean = "Website/URL"
            elif content_type == "📄 PDF Document":
                prompt = get_prompt("PDF Document")
                content_type_clean = "PDF Document"
            
            summary = generate_gemini_content(text_content, prompt)
            
            if summary:
                # Calculate statistics
                word_count = len(text_content.split())
                char_count = len(text_content)
                summary_words = len(summary.split())
                compression_ratio = (summary_words / word_count) * 100 if word_count > 0 else 0
                
                # Save to database
                save_summary(content_type_clean, source_url, file_name, summary, word_count, char_count, compression_ratio)
                
                # Display statistics
                st.markdown("""
                <div class="stats-container">
                    <div class="stat-item">
                        <span class="stat-value">{}</span>
                        <span class="stat-label">Original Words</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{}</span>
                        <span class="stat-label">Summary Words</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{:.1f}%</span>
                        <span class="stat-label">Compression</span>
                    </div>
                </div>
                """.format(word_count, summary_words, compression_ratio), unsafe_allow_html=True)
                
                # Display appropriate header based on content type
                if content_type == "🎥 YouTube Video":
                    st.markdown("### 🎥 Detailed Notes")
                elif content_type == "🌐 Website/URL":
                    st.markdown("### 🌐 Website Summary")
                elif content_type == "📄 PDF Document":
                    st.markdown("### 📄 Document Summary")
                
                st.markdown("""
                <div class="success-box">
                    ✅ Summary generated and saved successfully!
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="summary-result">
                    {}
                </div>
                """.format(summary), unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="error-box">
                    ⚠️ Could not generate summary
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="content-card">
        <h2>🤖 RAG-Powered Chatbot</h2>
        <p>Ask questions about your uploaded documents. The chatbot will provide answers based on the content you've added.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if there are any documents loaded
    rag_docs = get_rag_documents()
    
    if not rag_docs:
        st.warning("⚠️ No documents loaded yet! Please upload some content in the 'Content Summarizer' tab first.")
        st.info("💡 Upload YouTube videos, websites, or PDFs with the 'Add to RAG Chatbot' option enabled.")
    else:
        # Display loaded documents
        with st.expander(f"📚 View Loaded Documents ({len(rag_docs)} total)"):
            for i, (content_type, source_url, file_name, _, created_at) in enumerate(rag_docs):
                source_display = source_url or file_name or "Unknown"
                st.write(f"**{i+1}.** {content_type}: {source_display}")
                st.write(f"   Added: {created_at[:16]}")
        # Chat interface
        st.markdown("### 💬 Chat with your documents")
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="user-message">🧑‍💻 You: {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 Assistant: {message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        # Chat input
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What are the main points discussed in the uploaded content?",
            key="chat_input"
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Send Question", use_container_width=True):
                if user_question.strip():
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_question
                    })
                    with st.spinner("🤔 Thinking..."):
                        # Check if user wants detail
                        detail_phrases = [
                            'tell in detail', 'explain in detail', 'give me details', 'detailed answer', 'in detail', 'elaborate', 'full explanation'
                        ]
                        user_lower = user_question.lower()
                        wants_detail = any(phrase in user_lower for phrase in detail_phrases)
                        if wants_detail:
                            response = st.session_state.rag_system.generate_response(
                                user_question,
                                st.session_state.get("user_gemini_api_key")
                            )
                        else:
                            # Add instruction for concise answer
                            concise_query = user_question + "\n\nPlease answer concisely and only provide the required information. Do not elaborate or give extra details unless I ask for detail."
                            response = st.session_state.rag_system.generate_response(
                                concise_query,
                                st.session_state.get("user_gemini_api_key")
                            )
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    st.rerun()
                else:
                    st.error("Please enter a question.")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>Made with ❤️ using Streamlit 🚀, Gemini 🪐, and AI-Powered RAG 🤖</p>
    <p>Your summaries and documents are automatically saved for RAG-powered Q&A.</p>
</div>
""", unsafe_allow_html=True)
