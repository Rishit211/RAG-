import os
import requests
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default User-Agent setting
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "ChatBot/1.0 (+https://example.com; contact@example.com)")

class CustomWebLoader(WebBaseLoader):
    """
    A custom loader extending WebBaseLoader to allow custom headers.
    """
    def __init__(self, url: str, headers: dict = None):
        super().__init__(url)
        self.headers = headers or {}

    def _fetch_url(self, url: str) -> str:
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()  
        return response.text

def initialize_vector_store(documents):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)

    # Convert content into embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory='db')

  
    vector_store.persist()

    return vector_store

def retrieve_relevant_chunks(vector_store, user_query):
    retriever = vector_store.as_retriever()

  
    relevant_chunks = retriever.get_relevant_documents(user_query)
    return relevant_chunks

def generate_response(relevant_chunks, user_query, conversation_history):
   
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

   
    conversation_prompt = conversation_history + [
        ("user", user_query),
        ("assistant", "{context}")
    ]

    prompt_template = ChatPromptTemplate.from_messages(conversation_prompt)

    # Create a document chain for RAG
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Generate a response
    response = document_chain.invoke({"context": relevant_chunks, "input": user_query})
    
    # Append new interaction to the conversation history
    conversation_history.append(("assistant", response))
    
    return response, conversation_history

# Streamlit Application
st.set_page_config(page_title="Chat with Website (RAG Pipeline)", page_icon="🤖")
st.title("Chat with Website Using RAG Pipeline")

# Sidebar Input for URLs
with st.sidebar:
    st.header("Data Ingestion")
    url_input = st.text_area("Enter URLs (comma-separated):", placeholder="https://example.com, https://another-site.com")
    if st.button("Ingest Data"):
        urls = [url.strip() for url in url_input.split(",") if url.strip()]
        if urls:
            documents = []
            for url in urls:
                loader = CustomWebLoader(url, headers={"User-Agent": os.getenv("USER_AGENT")})
                documents.extend(loader.load())

            # Initialize vector store
            st.session_state.vector_store = initialize_vector_store(documents)
            st.success("Data ingested and stored in the vector database!")
        else:
            st.error("Please provide at least one valid URL.")

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Chat Interface
if "vector_store" in st.session_state:
    user_query = st.text_input("Type your question here:")
    if user_query:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(st.session_state.vector_store, user_query)

        # Generate response and update conversation history
        response, updated_history = generate_response(relevant_chunks, user_query, st.session_state.conversation_history)

        # Update conversation history in session state
        st.session_state.conversation_history = updated_history

        # Display response
        st.markdown(f"**Bot:** {response}")
else:
    st.info("Please ingest data first by providing URLs in the sidebar.")  