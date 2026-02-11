from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
#load the data
Data_Path = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=Data_Path)
#print("length of pdf file",len(documents))

#create the chunks

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

#vector embeddings

def get_embedding_model():
    model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model
embedding_model = get_embedding_model()

#store vector datasbase to FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)