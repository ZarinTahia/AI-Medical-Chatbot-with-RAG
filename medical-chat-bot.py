import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# Load .env (so HF_TOKEN is available)
# ----------------------------
load_dotenv(find_dotenv(), override=True)

# ----------------------------
# Config
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

# If v0.3 causes provider/router errors, switch to v0.2 or another hosted chat model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"


# ----------------------------
# Vectorstore (cached)
# ----------------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


# ----------------------------
# LLM (HF Inference chat)
# ----------------------------
def load_chat_llm(repo_id: str, hf_token: str):
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",                 # important for chat models
        huggingfacehub_api_token=hf_token,     # auth
        temperature=0.5,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)


# ----------------------------
# Helpers
# ----------------------------
format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))


# ----------------------------
# Streamlit app
# ----------------------------
def main():
    st.title("Ask Chatbot!")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN not found. Put it in your .env file as HF_TOKEN=xxxx and restart Streamlit.")
        st.stop()

    # Load vectorstore + retriever
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
    CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer from the context, say "I don't know." Do not make up an answer.
Do not provide anything outside the given context.

Context:
{context}

Question: {question}

Start the answer directly. No small talk.
""".strip()

    prompt_tmpl = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

    # LLM
    llm = load_chat_llm(HUGGINGFACE_REPO_ID, hf_token)

    # RAG chain (new LangChain LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_tmpl
        | llm
        | StrOutputParser()
    )

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    user_prompt = st.chat_input("Pass your prompt here")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            answer = rag_chain.invoke(user_prompt)
            docs = retriever.invoke(user_prompt)

            sources = []
            for d in docs:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", None)
                if page is not None:
                    sources.append(f"- {src} (page {page})")
                else:
                    sources.append(f"- {src}")

            result_to_show = answer
            if sources:
                result_to_show += "\n\n**Sources:**\n" + "\n".join(sources)

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        st.chat_message("assistant").markdown(result_to_show)
        st.session_state.messages.append({"role": "assistant", "content": result_to_show})


if __name__ == "__main__":
    main()
