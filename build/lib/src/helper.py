import os 
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()



OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')


#Extract the data from pdf 
def load_pdf(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


#Create text Chunks 
def text_split(data):
    extrated_data=load_pdf(data)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extrated_data)
    return text_chunks


embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)


def vectorestore(data):
    docs=text_split(data)
    vectorestor=FAISS.from_documents(docs,embeddings)
    vectorestor.save_local("Medical_Information")
    
def load_local():
    persisted_vectorstore=FAISS.load_local("Medical_Information",embeddings,allow_dangerous_deserialization=True)
    return persisted_vectorstore

