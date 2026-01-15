"""
Main Application UI
Streamlit interface for the RAG application.
"""
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.file_handler import FileProcessor
from src.rag_engine import RAGChatBot, get_answer

load_dotenv()

@st.cache_resource
def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )

@st.cache_resource
def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.3,
    )

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="UniBot - Your AI Tutor", layout="wide")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chatbot" not in st.session_state:
        st.session_state.rag_chatbot = None

    with st.sidebar:
        st.title("Proc0deBot Settings 🤖")
        
        api_key = None
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("API Key loaded from Streamlit Secrets.")
        except Exception:
            env_key = os.getenv("GOOGLE_API_KEY")
            if env_key:
                api_key = env_key
                st.success("API Key loaded from environment.")
        if not api_key:
            api_key = st.text_input("Enter Google API Key", type="password")
        if not api_key:
            st.warning("Please provide your Google API Key to continue.")

        st.divider()
        st.subheader("Upload Course Material")
        uploaded_files = st.file_uploader(
            "Upload course files",
            accept_multiple_files=True,
            type=["pdf", "docx", "xlsx", "png", "jpg", "jpeg"],
        )
        
        if st.button("Process Documents"):
            if not api_key:
                st.error("Please provide a valid Google API Key before processing.")
            elif not uploaded_files:
                st.warning("Please upload at least one supported file.")
            else:
                try:
                    with st.spinner("Analyzing course material..."):
                        all_chunks = []
                        processor = FileProcessor()
                        
                        for uploaded_file in uploaded_files:
                            suffix = os.path.splitext(uploaded_file.name)[1]
                            if not suffix:
                                suffix = ""
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            try:
                                chunks = processor.process_file(tmp_file_path)
                                all_chunks.extend(chunks)
                            finally:
                                if os.path.exists(tmp_file_path):
                                    os.remove(tmp_file_path)
                        
                        if not all_chunks:
                            st.error("Could not extract text from the provided files.")
                        else:
                            embeddings = get_embeddings(api_key)
                            llm = get_llm(api_key)
                            rag_chatbot = RAGChatBot(
                                api_key=api_key,
                                embeddings=embeddings,
                                llm=llm,
                            )
                            rag_chatbot.create_vector_store(all_chunks)
                            
                            st.session_state.rag_chatbot = rag_chatbot
                            st.session_state.vector_store = rag_chatbot.vector_store
                            
                            st.success("Processing Complete! You can now chat.")
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        st.divider()
        if st.button("Reset Conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # --- Main Chat Interface ---
    st.title("Proc0deBot: Chat with your Course 📚")

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ask a question about your course material..."):
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 2. Process Response
        if st.session_state.rag_chatbot:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get QA Chain
                        qa_chain = st.session_state.rag_chatbot.get_qa_chain()

                        # Generate Response with retry logic for rate limits
                        response = get_answer(qa_chain, user_input)
                        result = response["result"]
                        
                        st.markdown(result)
                        st.session_state.chat_history.append({"role": "assistant", "content": result})
                        
                        if "source_documents" in response:
                            with st.expander("🔍 Debug: View Source Text"):
                                for doc in response["source_documents"]:
                                    st.caption(f"Page {doc.metadata.get('page', 'N/A')}")
                                    st.text(doc.page_content)
                                    st.divider()
                                    
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        else:
            # Handle case where documents haven't been processed yet
            msg = "Please upload and process PDF documents in the sidebar first."
            with st.chat_message("assistant"):
                st.warning(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

if __name__ == "__main__":
    main()
