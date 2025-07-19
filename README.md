# 🚀 AI Content Summarizer 📝
A sophisticated content analysis solution empowered by the Google Gemini API and cutting-edge text processing technologies.

<img width="1917" height="1003" alt="image" src="https://github.com/user-attachments/assets/a88f1646-7f48-4726-895b-2df5dfafe2a7" />

<img width="1917" height="981" alt="image" src="https://github.com/user-attachments/assets/47ace9c2-364c-46c7-ba18-35a44da57f99" />
<img width="1912" height="966" alt="image" src="https://github.com/user-attachments/assets/1f2ec929-7a70-418d-80c2-c34c854a47b6" />

## 📖 Overview
The AI Content Summarizer is a powerful tool designed to transform lengthy content into concise, insightful summaries. Leveraging the Google Gemini API for summarization and a custom Retrieval-Augmented Generation (RAG) system for question-answering, it’s perfect for students, researchers, professionals, and content enthusiasts.
 - **live demo**: https://jasvinder21-ai-content-summarizer-main-mhre2b.streamlit.app/.

## ✨ Features

- **🎥 YouTube Video Summarizer**: Extracts transcripts and generates tailored summaries for educational, entertaining, or musical videos.
- **🌐 Website/URL Summarizer**: Scrapes and condenses web content into clear summaries.
- **📄 PDF Document Summarizer**: Processes and summarizes PDF files with detailed insights.
- **🤖 RAG-Powered Chatbot**: Ask questions about uploaded content with context-aware answers using a FAISS-based vector store and SentenceTransformer embeddings.
- **📚 Summary & Document History**: Stores summaries and documents in a local SQLite database for easy review and Q&A.
- **📊 Real-Time Statistics**: Displays word counts, summary lengths, and compression ratios.
- **🎨 Modern UI**: Custom-styled interface with gradient backgrounds, card layouts, and interactive elements.
- **🔑 API Integration**: Powered by your Google Gemini API key for seamless operation.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini API
- **RAG System**: SentenceTransformers (all-MiniLM-L6-v2), FAISS
- **Document Processing**: PyPDF2, PyMuPDF
- **Web Scraping**: BeautifulSoup, requests
- **Transcript Extraction**: youtube-transcript-api
- **Database**: SQLite
- **Environment Management**: python-dotenv
- **Additional Libraries**: NumPy, hashlib

## 📦 Installation
### Prerequisites

- 🐍 Python 3.8+
- 🔑 A Google Gemini API Key

## Setup

1. **Clone the Repository**
```bash
   git clone https://github.com/Jasvinder21/AI-Content-Summarizer.git
   cd ai-content-summarizer
```
2.**Install Dependencies**
```bash
   pip install -r requirements.txt
```
3. **Configure API Key** Create a *.env* file in the project root and add
```plain
   GOOGLE_API_KEY=your_api_key_here
```
4.**Run the Application**
```bash
   streamlit run main.py
```

## 📋 Requirements
Create a requirements.txt file with the following dependencies:
```txt
streamlit
google-generativeai
youtube-transcript-api
requests
beautifulsoup4
PyPDF2
PyMuPDF
sqlite3
python-dotenv
sentence-transformers
faiss-cpu
numpy
```
## 🚀 Usage

1.**Start the Application**
```bash
   streamlit run main.py
```
2. **Access the App** Open your browser and navigate to http://localhost:8501.or try the **live demo** at https://jasvinder21-ai-content-summarizer-main-mhre2b.streamlit.app/.

3. **Using the Application**
- Enter API Key: Input your Google Gemini API key in the sidebar.
- Select Content Type: Choose YouTube Video, Website/URL, or PDF Document.
- Provide Input: Enter a URL or upload a PDF file and click "Get Summary."
- Add to RAG Chatbot: Enable the option to store content for Q&A.
- Chat with Documents: Use the RAG Chatbot tab to ask questions about uploaded content.
- View History: Review summaries and loaded documents in the sidebar.

## ⚙️ Configuration

**API Key**
- Enter your Google Gemini API key in the sidebar to enable summarization and RAG functionality.
  
**RAG System:**
- Documents are chunked (1000 words with 200-word overlap) and stored in a FAISS vector store using SentenceTransformer embeddings.
- Supports context-aware Q&A with fallback to web search if no relevant content is found.

**Storage:** 
-Summaries and documents are saved in a local SQLite database (summarizer.db).

**Custom Styling:**
-Gradient backgrounds, card layouts, and responsive design enhance the user experience.

## 📁 Project Structure
```plain
ai-content-summarizer/
├── main.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API key)
├── README.md             # Project documentation
└── summarizer.db         # SQLite database for summaries
```

## 🔧 Key Functions

- **extract_transcript_details()**: Fetches YouTube video transcripts with language fallback support.
- **extract_website_text()**: Scrapes and cleans website content using BeautifulSoup.
- **extract_pdf_text()**: Extracts text from PDFs using PyMuPDF and PyPDF2..
- **generate_gemini_content()**: Generates summaries using the Gemini API.
- **AdaptiveRAG Class:**
- create_chunks(): Splits text into overlapping chunks for efficient retrieval.
- add_document(): Adds content to the RAG system with vector embeddings.
- retrieve_relevant_chunks(): Retrieves relevant document chunks using cosine similarity.
- generate_response(): Answers queries based on retrieved context with web search fallback.
-**Database Functions:**
- init_database(): Sets up SQLite tables for summaries and RAG documents.
- save_summary(): Stores summary metadata.
- save_rag_document(): Saves documents for RAG with deduplication using MD5 hashing.
- 
## 🎯 Use Cases

- **🎓 Educational Summaries**: Condense lectures, tutorials, or research papers.
- **🌍 Web Research**: Summarize articles or blog posts for quick insights.
- **📑 Document Review**: Extract key points from lengthy PDF reports.
- **🤖 Interactive Q&A**: Ask detailed questions about uploaded content for deeper understanding.

## 🔒 Privacy & Security

- **🖥️ Local Processing**:All processing occurs locally or via the Gemini API.
- **🔐 No External Sharing**: Content and summaries remain on your device.
- **💾 Local Storage**: Summaries and documents are stored in a local SQLite database.
- **🔑 API Key Security**: Your Gemini API key is only used for API calls and not stored in the app.
  
## 🤝 Contributing
Contributions are welcome! To contribute:

1. 🍴 Fork the repository.
2. 🌿 Create a feature branch (git checkout -b feature/new-feature).
3. 💾 Commit your changes (git commit -m 'Add new feature').
4. 🚀 Push to the branch (git push origin feature/new-feature).
5. 📬 Open a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit, Google Gemini API, and advanced RAG techniques.
- Thanks to the open-source community for libraries like SentenceTransformers, FAISS, and youtube-transcript-api.

## 📞 Support
For issues or questions:

1. Visit the Issues page.
2. Create a new issue with detailed information.
3. Ensure your Gemini API Key is correctly configured.

## 🚧 Roadmap

- Support for additional content types (e.g., DOCX, TXT).
- Enhanced RAG features (e.g., multi-document synthesis, advanced query handling).
- Exportable summary and chat history in various formats.
- Cloud deployment enhancements for scalability.
- Integration with additional AI models or APIs.
- 
**Built with 🖥️ Streamlit | Powered by 🪐 Gemini | Enhanced by 🤖 RAG**

