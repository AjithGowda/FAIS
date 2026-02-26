import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DB_FAISS_PATH = "faiss_db"
embedding = OllamaEmbeddings(model="llama3")
db = FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)


llm_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions based on the following retrieved documents.
Context: {context}
Question: {question}
Answer:""")


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = ({
    "context" : retriever | format_docs ,
    "question" : lambda x: x,
} | llm_prompt  | llm | StrOutputParser())

