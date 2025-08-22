import os
import re
import pathlib

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
import google.generativeai as genai
import logging

logging.basicConfig(
    filename="chat_log.txt",
    level=logging.INFO, 
    format="%(asctime)s-%(message)s"
)

# -----------------------------
# HTML background animation
# -----------------------------
particles_js = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: relative;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: 0; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 0;
    color: white;
  }
  </style>
</head>
<body>
  <div id=\"particles-js\"></div>
  <div class=\"content\"></div>
  <script src=\"https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js\"></script>
  <script>
    particlesJS("particles-js", {"particles":{"number":{"value": 100,"density":{"enable":true,"value_area":800}},"color":{"value":"#0175ff"},"shape":{"type":"circle","stroke":{"width":0,"color":"#000000"},"polygon":{"nb_sides":5},"image":{"src":"img/github.svg","width":100,"height":100}},"opacity":{"value":0.3,"random":false,"anim":{"enable":false,"speed":1,"opacity_min":0.2,"sync":false}},"size":{"value":4,"random":true,"anim":{"enable":true,"speed":1,"size_min":0.1,"sync":false}},"line_linked":{"enable":true,"distance":100,"color":"#ffffff","opacity":0.3,"width":1},"move":{"enable":true,"speed":1,"direction":"bottom","random":true,"straight":false,"out_mode":"out","bounce":false,"attract":{"false":true,"rotateX":600,"rotateY":1200}}},"interactivity":{"detect_on":"canvas","events":{"onhover":{"enable":true,"mode":"bubble, grab"},"onclick":{"enable":false,"mode":"push"},"resize":true},"modes":{"grab":{"distance":120,"line_linked":{"opacity":1}},"bubble":{"distance":150,"size": 8,"duration":2,"opacity":0.6,"speed":3},"repulse":{"distance":100,"duration":0.4},"push":{"particles_nb":4},"remove":{"particles_nb":2}}},"retina_detect":true});
  </script>
</body>
</html>
"""

# -----------------------------
# Env & API keys
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY is not set in your environment. Load it in a .env file or environment variable.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


def load_css(file_path: pathlib.Path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.info(f"Style file not found: {file_path}")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            # extract_text() can return None; guard it
            page_text = page.extract_text() or ""
            text += page_text
    return text


def chonky(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = FAISS.from_texts(text_chunks, embedding=embeddings)
    vs.save_local("faiss_index")


def get_doc_vectorstore():
    if not os.path.exists("faiss_index"):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# def get_query_vectorstore():
#     if not os.path.exists("query_index"):
#         return None
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return FAISS.load_local("query_index", embeddings, allow_dangerous_deserialization=True)


# def save_query_embedding(query: str):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if not os.path.exists("query_index"):
#         vs = FAISS.from_texts([query], embedding=embeddings)
#     else:
#         vs = get_query_vectorstore()
#         vs.add_texts([query])
#     vs.save_local("query_index")

# or the Past Queries sent by User in this session for finding answer
#         Past Queries sent by User in this session:
#        {query}
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a senior at Punjab Engineering College, you try to answer the questions as per the knowledge base provided. Try to generate response that are friendly. Avoid using "Alright", "Hey There", "So" and other filler words in the beginning of the sentence. 
        Answer in informal tone to sound friendly. Also include emojis in your response if relevent. 
        Try using knowledge base for finding the answer, but if the answer is not available in the context, reply with "Not enough information is available in the knowledge base trained on, but I can get an answer based on the Internet knowledge." and generate a response using Internet data.
        Try to strictly answer in 200 words. 
        Your Knowledge Base about PEC:
        {context}
        Question:
        {question}
        """
    )
    return create_stuff_documents_chain(model, prompt)


def user_input(user_question: str) -> str:
    new_db = get_doc_vectorstore()
    # query_db = get_query_vectorstore()

    docs = new_db.similarity_search(user_question) if new_db else []
    # past = query_db.similarity_search(user_question) if query_db else []

    chain = get_conversational_chain()
    # "query": past, 
    response = chain.invoke({"context": docs, "question": user_question})

    # save_query_embedding(user_question)
    return response


# def delete_query_index():
#     if os.path.exists("query_index"):
#         for root, dirs, files in os.walk("query_index", topdown=False):
#             for name in files:
#                 os.remove(os.path.join(root, name))
#             for name in dirs:
#                 os.rmdir(os.path.join(root, name))
#         os.rmdir("query_index")


def delete_faiss_index():
    has_any = os.path.exists("faiss_index") # or os.path.exists("query_index")
    if has_any:
        if os.path.exists("faiss_index"):
            for root, dirs, files in os.walk("faiss_index", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir("faiss_index")
        # delete_query_index()
        st.success("Cleaned up the cache")
    else:
        st.warning("Cache file doesn't exist")


# -----------------------------
# UI
# -----------------------------

def main():
    st.set_page_config(page_title="PEC GPT 2.0", page_icon="💬", layout="wide")

    css_path = pathlib.Path("style.css")
    load_css(css_path)

    # Background effect
    components.html(particles_js, height=370, scrolling=False)
    # Make the particles iframe full-screen and behind the app
    st.markdown(
        """
        <style>
        /* Target the Streamlit iframe that holds the particles component */
        iframe[title="st.iframe"] {
            position: fixed !important;
            inset: 0;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 1 !important;        /* push behind Streamlit UI */
            pointer-events: auto !important;/* keep UI clickable */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Session state for messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
        # delete_query_index()

    # Header & intro
    st.header("PEC GPT 2.0", divider="red")
    st.markdown(
        '<div class="intro">Welcome to PEC GPT 2.0! designed by PEC ACM to help you with your queries based on PEC Chandigarh (Deemed to be University) \n\n It will try to answer your queries with it\'s best ability.',
        unsafe_allow_html=True,
    )

    # Display chat history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Sidebar
    with st.sidebar:
        st.markdown('<img src="https://avatars.githubusercontent.com/u/54832562?s=280&v=4" width=80 >', unsafe_allow_html=True)
        st.header("PEC GPT 2.0", divider="red")
        # st.subheader("Upload PDF Documents")
        # pdf_docs = st.file_uploader("Pick a pdf file", type="pdf", accept_multiple_files=True)

        # if pdf_docs and st.button("Process Documents", key="green"):
        #     with st.spinner("Processing"):
        #         raw_text = get_pdf_text(pdf_docs)
        #         if raw_text.strip():
        #             text_chunks = chonky(raw_text)
        #             if text_chunks:
        #                 get_vectorstore(text_chunks)
        #                 st.markdown('<div class="donepdf">Done</div>', unsafe_allow_html=True)
        #             else:
        #                 st.warning("No text chunks were created from the uploaded PDFs.")
        #         else:
        #             st.warning("No text could be extracted from the uploaded PDFs.")

        # if not pdf_docs:
        #     st.markdown('<div class="uppdf">Please upload a PDF file to start</div>', unsafe_allow_html=True)

        st.markdown('<div class="blanki"></div>', unsafe_allow_html=True)
        st.markdown('<div class="luvacm">Made with ❤️ by PEC ACM </div>', unsafe_allow_html=True)
        st.link_button("View the source code", "https://github.com/Ya-Tin/PDFQueryChatLM.git")

        # if st.button("Reset Bot Memory", key="red"):
        #     delete_faiss_index()

        # if st.button("Stop App", key="red2"):
        #     # delete_query_index()
        #     os._exit(0)

    # Chat input
    user_question = st.chat_input("Input your Query here and press Enter")
    logging.info(f"User asked: {user_question}")

    if user_question:
        st.session_state["messages"].append({"role": "user", "content": user_question})
        st.chat_message("user").markdown(user_question)

        with st.spinner("Generating response..."):
            response = user_input(user_question)

        # Remove the fallback preface if present
        unwanted_line_pattern = r"Not enough information is available in the documents provided, but I can get an answer based on the Internet knowledge\."
        response = re.sub(unwanted_line_pattern, "", response)

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)


if __name__ == "__main__":
    main()
