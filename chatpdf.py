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
import logging

st.set_page_config(page_title="Chat PDF", layout="wide")

# Load environment variables
load_dotenv()
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Configure Google Generative AI
if not GOOGLE_GENAI_API_KEY:
    raise EnvironmentError("GOOGLE_GENAI_API_KEY is not set. Please check your environment variables.")
genai.configure(api_key=GOOGLE_GENAI_API_KEY)

# Set the application credentials
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
else:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS is not set. Please check your environment variables.")

# Logging setup for debugging
logging.basicConfig(level=logging.INFO)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Generate vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logging.error(f"Error in creating vector store: {e}")
        st.error(f"Failed to create vector store: {e}")

def get_conversational_chain():
    """Set up the conversational chain for QA."""
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, just say, "answer is not available in the context." Do not provide incorrect answers.\n\n
        Context:\n {context}\n
        Question: {question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error in setting up conversational chain: {e}")
        st.error(f"Failed to set up conversational chain: {e}")

def user_input(user_question):
    """Process user input and fetch response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Allow dangerous deserialization for trusted sources
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        logging.error(f"Error in processing user input: {e}")
        st.error(f"Failed to process question: {e}")

def main():
    """Main Streamlit application."""
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed successfully!")
                except Exception as e:
                    logging.error(f"Error during PDF processing: {e}")
                    st.error(f"Failed to process PDF files: {e}")

if __name__ == "__main__":
    main()
