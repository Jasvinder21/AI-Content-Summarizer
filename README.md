# 🚀 AI Content Summarizer 📝
A sophisticated content analysis solution empowered by the Google Gemini API and cutting-edge text processing technologies.

This application is built upon advanced natural language processing (NLP) and state-of-the-art machine learning frameworks, harnessing the Gemini API’s robust contextual analysis capabilities to deliver precise and contextually relevant summaries, optimized for professional and academic use.
<img width="1917" height="967" alt="image" src="https://github.com/user-attachments/assets/a2bd9369-934c-4765-8948-735d026e1222" />
<img width="1913" height="971" alt="image" src="https://github.com/user-attachments/assets/fc3cdbab-8dc5-408d-bbc1-1a9cb4359acf" />

## 📖 Overview
The AI Content Summarizer is a Streamlit-based web application that allows users to summarize various content types using advanced AI capabilities. Leveraging the Gemini API, it provides accurate and engaging summaries, making it ideal for students, professionals, and content enthusiasts.

## ✨ Features

-**🎥 YouTube Video Summarizer8**: Extract transcripts and create summaries (Educational, Entertaining, or Musical).
-**🌐 Website/URL Summarizer**: Scrape and summarize web content instantly.
-**📄 PDF Document Summarizer**: Process and summarize PDF files effortlessly.
-**📚 Summary History**: Review and manage recent summaries with timestamps.
-**📊 Real-Time Statistics**: View word counts and compression ratios.
-**🎨 Modern UI**: Sleek, responsive interface with custom styling.
-**🔑 API Integration**: Customizable with your Google Gemini API Key.

## 🛠️ Technology Stack

-**Frontend**: Streamlit
-**AI Model**: Google Gemini API
-**Document Processing**: PyPDF2, PyMuPDF
-**Web Scraping**: BeautifulSoup, requests
-**Transcript Extraction**: youtube-transcript-api
-**Database**: SQLite
-**Environment**: python-dotenv

## 📦 Installation
### Prerequisites

- 🐍 Python 3.8+
- 🔑 A Google Gemini API Key

## Setup

1. **Clone the Repository**
```bash
   git clone https://github.com/your-username/ai-content-summarizer.git
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
```
## 🚀 Usage

1.**Start the Application**
```bash
   streamlit run main.py
```
2. **Access the App** Open your browser and navigate to http://localhost:8501.
3. **Using the Application**
- Enter your Gemini API Key in the sidebar.
- Select a content type (YouTube Video, Website/URL, or PDF Document).
- Provide the input (URL or file upload) and click "Get Summary".
- View the summary and explore the history in the sidebar.

## ⚙️ Configuration

**API Key**
- Enter your Google Gemini API Key in the sidebar to enable the app.

**Processing Options**
- **YouTube Types**: Choose Educational, Entertaining, or Musical summaries.
- **Summary Storage**: Automatically saved to a local SQLite database.

## 📁 Project Structure
```plain
ai-content-summarizer/
├── main.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── summarizer.db         # SQLite database for summaries
```

## 🔧 Key Functions

-**extract_transcript_details()**: Fetches YouTube video transcripts.
-**extract_website_text()**: Scrapes and cleans website content.
-**extract_pdf_text()**: Extracts text from PDF files.
-**generate_gemini_content()**: Generates summaries using the Gemini API.
-**save_summary()**: Stores summaries in the database.

## 🎯 Use Cases

-**🎓 Educational Summaries**: Summarize lectures or tutorials.
-**🌍 Web Research**: Quickly grasp key points from articles.
-**📑 Document Review**: Condense lengthy PDF reports.
-**🎵 Lyrics Extraction**: Extract song lyrics from music videos.

## 🔒 Privacy & Security

-**🖥️ Local Processing**: Runs locally with your API key.
-**🔐 No External Sharing**: Content stays on your device.
-**💾 Local Storage**: Summaries saved in a local SQLite database.

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

- Built with ❤️ using Streamlit, Gemini API, and AI-powered content analysis.
- Thanks to the open-source community for supporting libraries.

## 📞 Support
For issues or questions:

1. Visit the Issues page.
2. Create a new issue with detailed information.
3. Ensure your Gemini API Key is correctly configured.

## 🚧 Roadmap

- Support for additional content types (e.g., DOCX, TXT).
- Advanced summarization options (e.g., bullet points, key phrases).
- Exportable summary history.
- Cloud deployment support.

**Built with 🖥️ Streamlit | Powered by 🪐 Gemini**
