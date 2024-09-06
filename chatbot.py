#!/usr/bin/env python3


"""
This script lays out different methods for ai-chatbot
"""

import os
import sys
from typing import List
import openai
import argparse

    
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
_ = load_dotenv(dotenv_path, override=True) # read local .env file
#print(f"Loading .env file from: {dotenv_path}")

openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'

def document_loading() -> List[Document]:
    # currently only one pdf is available
    loader = PyPDFLoader("/Users/arushi/Downloads/MachineLearning-Lecture01.pdf")
    pages = loader.load()
    #print(len(pages))
    page = pages[0]
    #print(page.page_content[0:500])
    #print(page.metadata)
    return pages

def document_splitter(pages):
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    total_splits = r_splitter.split_documents(pages)
    #print(splits)
    return total_splits
        
def create_embedding_and_store_in_vector_db(splits):
    print(len(splits))

    try: 
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        print(vectordb._collection.count())
        question = "is there an email i can ask for help"
        docs = vectordb.similarity_search(question,k=3)
        print(docs[0].page_content)
        #vectordb.persist() manual call not needed. Automatically persisted

    except Exception as e:
        print(e)

def qa_retrieval_chain():
    llm_name = "gpt-3.5-turbo"
    persist_directory = 'docs/chroma/'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    #print(vectordb._collection.count())
    
    question = "What are main topics for this class?"
    docs = vectordb.similarity_search(question,k=3)
    print(docs[0])

    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    result = qa_chain({"query": question})
    print(result["result"])

def retrieval_qa_chain_with_prompt(question: str):
    
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    print(result["result"])

def conversational_retrieval_chain_with_memory(question: str):

    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever=vectordb.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    print(result['answer'])


if __name__ == "__main__":
    # pages = document_loading()
    # splits = document_splitter(pages)
    # create_embedding_and_store_in_vector_db(splits[:10])
    # qa_retrieval_chain()

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description="Type in your question")
    parser.add_argument('-q', '--question', type=str, required=True, help="Question")
    args = parser.parse_args()
    
    #retrieval_qa_chain_with_prompt(args.question)
    conversational_retrieval_chain_with_memory(args.question)
