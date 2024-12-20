import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from functools import lru_cache
import time
from typing import List, Tuple
import numpy as np
import joblib

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Cache save/load utility
CACHE_DIR = "cache_data"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def save_cache_to_disk(cache_name: str, data):
    """Save the cache data to the disk using joblib."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    joblib.dump(data, cache_path)

def load_cache_from_disk(cache_name: str):
    """Load the cache data from disk if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    return None

# Create a single instance of embeddings to reuse
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Cache for PDF text extraction
@lru_cache(maxsize=100)
def cached_page_extraction(page_content: str) -> str:
    """Cache the text extraction from PDF pages"""
    cached_data = load_cache_from_disk('page_extraction')
    if cached_data:
        return cached_data.get(page_content, page_content)  # Use cached if exists
    return page_content

def get_pdf_text(pdf_docs):
    """Extract text from PDFs with caching"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # Convert page content to string for caching
            page_content = page.extract_text()
            # Use cached version if available
            cached_text = cached_page_extraction(page_content)
            text += cached_text
    save_cache_to_disk('page_extraction', {page_content: cached_text})  # Save updated cache
    return text

def batch_process_chunks(chunks: List[str], batch_size: int = 5) -> List[str]:
    """Process text chunks in batches"""
    processed_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        processed_chunks.extend(batch)
        time.sleep(1)  # Add delay between batch processing
    return processed_chunks

def get_text_chunks(text: str) -> List[str]:
    """Split text into chunks with batch processing"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    # Process chunks in batches
    return batch_process_chunks(chunks)

# Cache for embeddings
@lru_cache(maxsize=1000)
def cached_embedding(text_chunk: str) -> Tuple[float]:
    """Cache embeddings for text chunks"""
    try:
        embeddings = get_embeddings()
        time.sleep(1)  # Add delay between embedding calls
        return tuple(embeddings.embed_query(text_chunk))
    except Exception as e:
        print(f"Error in embedding: {e}")
        time.sleep(60)  # Wait longer if we hit an error
        return tuple(embeddings.embed_query(text_chunk))

def batch_create_embeddings(text_chunks: List[str], batch_size: int = 3):  # Reduced batch size
    """Create embeddings in batches"""
    all_embeddings = []
    embeddings = get_embeddings()  # Reuse same embeddings instance

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            # Try to get embeddings from cache first
            batch_embeddings = [cached_embedding(chunk) for chunk in batch]
            all_embeddings.extend(batch_embeddings)
            time.sleep(2)  # Add delay between batches
        except Exception as e:
            print(f"Error in batch embedding: {e}")
            time.sleep(60)  # Wait longer if we hit an error
            batch_embeddings = [cached_embedding(chunk) for chunk in batch]
            all_embeddings.extend(batch_embeddings)

    save_cache_to_disk('embeddings', all_embeddings)  # Save updated cache
    return all_embeddings

def get_vector_store(text_chunks: List[str]):
    """Create vector store with batched and cached embeddings"""
    try:
        embeddings = get_embeddings()  # Reuse same embeddings instance
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_vector_store")
    except Exception as e:
        print(f"Error in vector store creation: {e}")
        time.sleep(60)  # Wait if we hit an error
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_vector_store")

# Cache for conversation chain
@lru_cache(maxsize=1)
def get_conversational_chain():
    """Cache the conversation chain"""
    prompt_template = """
    Answer the question using the provided context. Ensure the response is comprehensive and beautifully formatted in Markdown as follows:
    - **Headings:** Use `###` for headings and only headings, not for normal text.
    - **Subheadings:** Use `####` for subheadings and only subheadings, not for normal text.
    - **Normal Text:** Write normal text without any special characters.
    - **Details:** Use bullet points (`- `) or numbered lists (`1. `) for clarity.
    - **New Lines:** Use double new lines (`\\n\\n`) to separate paragraphs and sections.
    - **Emphasis:** Use `**` for bold and `_` for italics to highlight important information.

    If the answer is not available in the provided context, clearly state:
    "The answer is not available in the provided context."

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Cache for similarity search results
@lru_cache(maxsize=100)
def cached_similarity_search(question: str):
    """Cache similarity search results"""
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_vector_store", embeddings, allow_dangerous_deserialization=True)
        return new_db.similarity_search(question)
    except Exception as e:
        print(f"Error in similarity search: {e}")
        time.sleep(60)  # Wait if we hit an error
        return new_db.similarity_search(question)

def user_input(user_question: str):
    """Process user input with caching and stream responses beautifully formatted."""
    try:
        # Use cached similarity search
        docs = cached_similarity_search(user_question)
        # Use cached conversation chain
        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        # Prepare streaming response
        response_text = response["output_text"]

        # Markdown Formatting Fixes
        formatted_response = response_text.strip()  # Remove extra spaces
        formatted_response = formatted_response.replace("\n", "\n\n")  # Ensure Markdown interprets new lines
        formatted_response = formatted_response.replace("* ", "- ")  # Normalize bullet points

        st.write("### Response:")  # Heading for the response
        placeholder = st.empty()  # Placeholder for streaming content

        # Stream by paragraphs or sentences
        paragraphs = formatted_response.split("\n\n")
        streamed_text = ""
        for paragraph in paragraphs:
            streamed_text += "\n\n"
            for word in paragraph.split():
                streamed_text += word + " "
                placeholder.markdown(streamed_text, unsafe_allow_html=False)
                time.sleep(0.01)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        time.sleep(60)  # Wait if we hit an error

def main():
    st.set_page_config("Multiple PDF Chatbot")
    st.header("Chat with PDFs")
   
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
   
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
       
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    # Process PDFs with caching and batching
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    time.sleep(60)  # Wait if we hit an error

if __name__ == "__main__":
    main()
