import os
from pymongo import MongoClient
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB connection
mongo_client = MongoClient("mongodb+srv://naman0908be21:boi@cluster0.wm6dn.mongodb.net/")
db = mongo_client['chat_database']
collection = db['chats']

def save_chat_to_db(user_message, assistant_response):
    chat = {
        "user_message": user_message,
        "assistant_response": assistant_response
    }
    collection.insert_one(chat)

def get_chat_history():
    return list(collection.find())

# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "location based assistant"}]
    st.experimental_rerun()  # Add this line to refresh the Streamlit app

def get_answer_from_gemini(prompt):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro",
                                       client=genai,
                                       temperature=0.3,
                                       )
        response = model.generate(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error occurred while getting response from Gemini API: {e}")
        return "Sorry, an error occurred while fetching the answer."

def user_input(user_question):
    # Ensure user_question is not empty
    if not user_question:
        st.warning("Please enter a valid question.")
        return {"output_text": []}

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    if os.path.exists("faiss_index/index.faiss"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        st.warning("FAISS index not found. Please upload PDFs and process them.")
        return {"output_text": []}

    # Search for similar documents
    try:
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error occurred while searching for similar documents: {e}")
        return {"output_text": []}

    # Get conversational chain
    chain = get_conversational_chain()

    # Generate response from context
    try:
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response_text = ''.join(response['output_text'])

        if "answer is not available in the context" in response_text.lower():
            # Use Gemini API as a fallback
            fallback_response = get_answer_from_gemini(user_question)
            response_text = fallback_response

        return {"output_text": [response_text]}

    except Exception as e:
        st.error(f"Error occurred while generating response: {e}")
        return {"output_text": []}

def main():
    # Custom CSS for styling
    st.set_page_config(
        page_title="CHITRAGUPTA",
        page_icon="🤖"
    )

    st.markdown(
        """
        <style>
        /* Custom Streamlit CSS */
        body {
            background-color: #121212;  /* Dark background color */
            color: #ffffff;             /* White text color */
            font-family: Arial, sans-serif;  /* Font family */
        }
        .stButton>button {
            background-color: #1f77b4;  /* Button background color */
            color: #ffffff;             /* Button text color */
            border-color: #1f77b4;      /* Button border color */
        }
        .stButton>button:hover {
            background-color: #1a5e91;  /* Button background color on hover */
            color: #ffffff;             /* Button text color on hover */
            border-color: #1a5e91;      /* Button border color on hover */
        }
        .stProgress>div>div>div>div {
            background-color: #1f77b4;  /* Progress bar color */
        }
        .chat-box {
            background-color: #2a2a2a;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .user-msg {
            text-align: right;
            color: #e1e1e1;
        }
        .assistant-msg {
            text-align: left;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Backend PDF upload and processing
    pdf_docs = ["Chitragupta.pdf"]  # Add your PDF paths here
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    if not st.session_state.pdf_processed:
        if pdf_docs:  # Ensure there are files to process
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully.")
                st.session_state.pdf_processed = True

    st.title("CHITRAGUPTA AI")
    st.write("Welcome to the chat!")

    if st.session_state.pdf_processed:
        if st.button('Clear Chat History'):
            clear_chat_history()
            
    if st.button('Show Chat History'):
        chat_history = get_chat_history()
        if chat_history:
            for chat in chat_history:
                st.markdown(
                    f'<div class="chat-box assistant-msg">User: {chat["user_message"]}<br>Assistant: {chat["assistant_response"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.write("No chat history found.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Your local city assistant for Patiala, Zirakpur, Kurukshetra and Chandigarh"}]

    for message in st.session_state.messages:
        role_class = 'user-msg' if message["role"] == "user" else 'assistant-msg'
        st.markdown(
            f'<div class="chat-box {role_class}">{message["role"]}: {message["content"]}</div>',
            unsafe_allow_html=True
        )

    user_input_text = st.text_input('Your message')
    if user_input_text:
        st.session_state.messages.append(
            {"role": "user", "content": user_input_text})
        st.markdown(
            f'<div class="chat-box user-msg">user: {user_input_text}</div>',
            unsafe_allow_html=True
        )

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = user_input(user_input_text)
            full_response = ''
            for item in response['output_text']:
                full_response += item
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
            st.markdown(
                f'<div class="chat-box assistant-msg">assistant: {full_response}</div>',
                unsafe_allow_html=True
            )
            save_chat_to_db(user_input_text, full_response)

if __name__ == "__main__":
    main()
