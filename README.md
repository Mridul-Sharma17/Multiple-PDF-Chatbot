# Multiple PDF Chatbot

A powerful chatbot application that allows users to interact with multiple PDF documents using advanced natural language processing and the Gemini Pro model.

## Features

- Upload and process multiple PDF documents
- Extract and chunk text content intelligently
- Generate vector embeddings for efficient search
- Interactive Q&A with uploaded documents
- Beautiful streaming responses in Markdown format
- Caching system for improved performance

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini Pro model

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd Multiple-PDF-Chatbot
```
## Create Virtual Environment

### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Google API key
```
GOOGLE_API_KEY=your_api_key_here
```

## Libraries and Packages Used

- **streamlit**: Web application framework for the user interface
- **PyPDF2**: PDF document processing
- **langchain**: Framework for language model interactions
- **google.generativeai**: Google's Generative AI API
- **FAISS**: Vector storage and similarity search
- **python-dotenv**: Environment variable management
- **joblib**: Caching and persistence
- **numpy**: Numerical computations
- **typing**: Type hints for better code documentation

## Usage

1. Start the application
```bash
streamlit run app.py
```

2. Upload PDF documents using the sidebar
3. Click "Submit & Process" to process the documents
4. Ask questions in the text input field
5. View beautifully formatted responses in real-time

## Key Functions

- `get_pdf_text()`: Extracts text from PDF documents
- `get_text_chunks()`: Splits text into manageable chunks for processing
- `batch_create_embeddings()`: Creates vector embeddings in batches with caching
- `get_vector_store()`: Manages vector storage for similarity search
- `get_conversational_chain()`: Sets up the Q&A chain with custom prompts
- `user_input()`: Processes user questions and streams formatted responses
- `main()`: Orchestrates the application flow and UI components

## Performance Optimizations

- Intelligent caching system for text extraction and embeddings
- Batch processing to manage API rate limits
- Error handling with automatic retries
- Efficient vector similarity search
- Streaming responses for better user experience
