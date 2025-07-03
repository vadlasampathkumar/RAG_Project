
# 📚 Indian Budget RAG Chatbot

This project is an **end-to-end Retrieval-Augmented Generation (RAG)** pipeline that enables users to interact with the Indian Budget 2024–25 speech using natural language — **in multiple languages and through voice**.

Built using **Google Gemini**, **LangChain**, **Pinecone**, and **Streamlit**, it allows users to upload a PDF of the budget, ask questions in any supported language (typed or spoken), and receive answers with references to the original document.

---

## 🚀 Key Features

- 🧠 Retrieval-Augmented Generation using **Gemini + Pinecone**
- 🗣️ **Voice-based Q&A** with automatic speech recognition and TTS (gTTS)
- 🌐 **Multilingual Support** using `deep-translator` and Gemini
- 📄 Accepts any **Budget PDF** (tested with 2024-25 Budget)
- 🔍 Returns **source snippets** for transparency and traceability

---

## 🛠️ Project Structure

```
.
├── budget_speech.pdf           # Budget 2024–25 Speech PDF (sample)
├── streamlit_app.py            # Standard RAG chatbot (text-based)
├── streamlit_voice.py          # Voice-based chatbot
├── streamlit_translator.py     # Multilingual + voice chatbot
├── RAG_end_to_end.ipynb        # Notebook version of the complete pipeline
├── required.txt                # Dependencies
└── .env                        # API keys for Gemini and Pinecone
```

---

## 📦 Setup Instructions

### 1. Clone and Install Requirements

```bash
pip install -r required.txt
```

### 2. Add API Keys

Create a `.env` file with the following:

```bash
GEMINI_API_KEY=your_gemini_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### 3. Run the Apps

#### Text-Based Chatbot

```bash
streamlit run streamlit_app.py
```

#### Voice-Based Chatbot

```bash
streamlit run streamlit_voice.py
```

#### Multilingual Voice Chatbot

```bash
streamlit run streamlit_translator.py
```

---

## 💡 How It Works

1. **Load PDF** using LangChain's `PyPDFLoader`
2. **Chunk the text** with `RecursiveCharacterTextSplitter`
3. **Generate embeddings** via `GoogleGenerativeAIEmbeddings`
4. **Store vectors** in **Pinecone**
5. **Query Gemini** through a `RetrievalQA` chain using a custom prompt
6. **Translate**, **Speak**, or **Render** results depending on the interface

---

## 🌍 Supported Languages (Multilingual App)

- Hindi, Telugu, Tamil, Kannada, Marathi, Gujarati, Bengali, Punjabi, Malayalam, English

---

## 📌 Prompt Engineering

A carefully crafted prompt ensures Gemini answers **factually**, **politely**, and based solely on the uploaded Budget document. It avoids hallucinations and includes a fallback message if an answer isn't found.

---

## 🙏 Acknowledgments

- Developed under guidance of **Omkar Sir**
- Uses data from the **Union Budget 2024–2025 Speech**
- Powered by **LangChain**, **Google Gemini**, **Pinecone**, and **Streamlit**

---

## 📚 Use Cases

- Budget analysis by students, researchers, or journalists
- Natural language search through large financial documents
- Local-language accessibility for policy understanding

---

## 🔐 License

This project is for educational and demonstration purposes only.
