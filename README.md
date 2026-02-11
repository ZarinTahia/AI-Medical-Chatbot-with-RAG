# AI-Medical-Chatbot-with-RAG 

## Overview

This project is an AI-powered Medical Question Answering Chatbot built using Retrieval-Augmented Generation (RAG). The system retrieves relevant medical knowledge from a document knowledge base and uses a language model to generate accurate, context-aware answers.

The chatbot is designed to answer medical-related questions only, helping users get reliable information grounded in trusted medical documents rather than relying purely on generative AI.

## Tech Stack

Languages & Frameworks: Python and Streamlit

AI / ML: HuggingFace Transformers, Sentence Transformers, LangChain

Vector Database: FAISS

## Project Structure

```
AI-Medical-Chatbot-with-RAG/
│
├── connect_memory_with_llm.py   # connection establish
├── medical-chat-bot.py          # Main Streamlit chatbot app
├── memory_llm.py                # Memory-enabled LLM pipeline
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── data/                        # Medical documents
```


## Run the application
streamlit run medical-chat-bot.py
