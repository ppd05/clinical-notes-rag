# 🏥 Clinical Notes RAG System

> AI-powered Retrieval-Augmented Generation system for processing and querying clinical notes using Google's Gemini 2.0 Flash-Lite, LangChain, and FAISS.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- 📄 **Unstructured to Structured**: Automatically converts clinical notes into structured JSON format
- 🔍 **Semantic Search**: FAISS-powered vector database for fast, relevant information retrieval
- 📝 **AI Summarization**: Generate comprehensive summaries of patient records
- 💬 **Q&A System**: Ask natural language questions about clinical notes with patient filtering
- ⚡ **Rate Limiting**: Built-in protection (2 patients per session, 5-minute cooldown)
- 🎨 **Interactive UI**: Beautiful, intuitive Streamlit interface
- 🆓 **Cost-Effective**: Uses free HuggingFace embeddings + Gemini API


## 🏗️ Architecture

```
Clinical Notes (.txt)
        ↓
    Gemini 2.0 Flash-Lite (Structure Extraction)
        ↓
    Structured Data (JSON)
        ↓
    RecursiveTextSplitter (Chunking)
        ↓
    HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
    FAISS Vector Store
        ↓
    RAG Pipeline (Query + Context Retrieval)
        ↓
    Gemini 2.0 Flash-Lite (Answer Generation)
```

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get it free](https://makersuite.google.com/app/apikey))
- 2GB+ RAM (for embedding model)

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ppd05/clinical-notes-rag.git
cd clinical-notes-rag
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*First-time installation takes 2-3 minutes (downloads ~500MB for models)*

### 4. Configure API Key

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Processing Clinical Notes

1. **Upload Files**: Click "Browse files" and select `.txt` clinical notes
2. **Process**: Click "Process Clinical Notes" button
3. **Wait**: Processing takes 20-30 seconds per file

### Viewing Results

1. **Select Patient**: Choose from dropdown menu
2. **View Data**: See structured JSON extraction
3. **Generate Summary**: Click "Generate Summary" for AI overview

### Asking Questions

1. **Enter Question**: Type your query (e.g., "What medications were prescribed?")
2. **Filter**: Optionally select specific patient or search all
3. **Search**: Click "Search" to get AI-powered answer

## 🎯 Rate Limiting

Built-in protection against abuse:

- **Limit**: 2 patients per session
- **Cooldown**: 15 minutes after limit reached
- **Tracking**: Real-time usage display in sidebar

To reset: Refresh the page after cooldown period.

## 📁 Project Structure

```
clinical-notes-rag/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── DEPLOYMENT_PROTECTION.md # Security & deployment guide
│
├── .streamlit/
│   └── config.toml          # Streamlit configuration
│
└── sample_notes/            # Example clinical notes
    ├── patient1.txt
    └── patient2.txt
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Rate Limiting Settings

Edit in `app.py`:

```python
MAX_PATIENTS_PER_SESSION = 2  # Files per session
COOLDOWN_MINUTES = 15          # Wait time in minutes
```

### Embedding Model

Currently uses: `sentence-transformers/all-MiniLM-L6-v2`

To change, edit in `app.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="your-preferred-model",
    model_kwargs={'device': 'cpu'}
)
```

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in app settings:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

## 🛡️ Security

- ✅ API keys stored in environment variables (never in code)
- ✅ `.env` file in `.gitignore`
- ✅ Rate limiting enabled by default
- ✅ Streamlit secrets for deployment
- ✅ No hardcoded credentials

**Never commit `.env` file to version control!**

## 🧪 Example Clinical Note Format

```text
Patient ID: P12345
Date: 2024-10-15

Chief Complaint: Patient presents with persistent cough and fever for 5 days.

History: 45-year-old male with no significant medical history.

Examination: Temperature 100.8°F, BP 128/82

Diagnosis: Community-acquired pneumonia

Treatment Plan:
- Prescribed Amoxicillin 500mg TID for 7 days
- Acetaminophen for fever management
- Rest and increased fluid intake

Recommendations: Follow-up in 1 week
```

## 🔍 Technical Details

### Models Used

- **LLM**: Google Gemini 2.0 Flash-Lite
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (CPU)

### Text Processing

- **Splitter**: RecursiveCharacterTextSplitter
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

### Supported File Types

- Plain text (`.txt`) files only
- UTF-8 encoding

## 📊 API Limits (Free Tier)

**Gemini API:**
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per day

**HuggingFace Embeddings:**
- Unlimited (runs locally)

Monitor usage: https://ai.dev/usage

## 🐛 Troubleshooting

### "API Key not found"
- Ensure `.env` file exists in project root
- Verify `GEMINI_API_KEY` is set correctly
- Check virtual environment is activated

### "Quota exceeded"
- Wait for quota reset (check https://ai.dev/usage)
- Rate limiting prevents most quota issues
- Consider upgrading to paid tier

### "Module not found"
- Reinstall dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated

### Rate limit not working
- Clear browser cache
- Refresh the page
- Check session state in app

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. 

- **Not for Production**: Not intended for real clinical use
- **HIPAA Compliance**: Not HIPAA compliant - do not use with real patient data
- **No Medical Advice**: Does not provide medical advice
- **Accuracy**: AI-generated content may contain errors
- **Liability**: Use at your own risk

Always ensure proper data security and compliance when handling medical information.


## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for the LLM API
- [LangChain](https://python.langchain.com/) for RAG framework
- [HuggingFace](https://huggingface.co/) for embedding models
- [Streamlit](https://streamlit.io/) for the UI framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Anthropic,Claude](https://www.anthropic.com/) for the streamlit UI design and code debugging.

## 📈 Roadmap

- [ ] Support for PDF files
- [ ] Multi-language support
- [ ] Export functionality (PDF, DOCX)
- [ ] Advanced analytics dashboard
- [ ] Integration with EHR systems
- [ ] Voice input support

## ⭐ Star History

If you find this project useful, please give it a star! ⭐

---

**Built with ❤️ using Google Gemini 2.0 Flash-Lite, LangChain, and Streamlit**

