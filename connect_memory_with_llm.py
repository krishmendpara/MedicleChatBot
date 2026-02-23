# connect_memory_with_llm.py

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# =====================
# CONFIG
# =====================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"


# =====================
# LOAD LLM
# =====================

print("Connecting to Groq...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=512,
    api_key=GROQ_API_KEY
)


# =====================
# LOAD VECTOR DB
# =====================

print("Loading embeddings...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS...")

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# =====================
# PROMPT
# =====================

prompt = ChatPromptTemplate.from_template("""
You are a professional medical assistant.

Answer using ONLY the provided context.

Rules:
- Answer in 4to 5 lines maximum
- Be short and concise
- Do not explain too much
- Do not add extra information

Context:
{context}

Question:
{input}

Short Answer:
""")


# =====================
# CREATE CHAIN
# =====================

combine_docs_chain = create_stuff_documents_chain(llm, prompt)

qa_chain = create_retrieval_chain(
    db.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain
)

print("Medical AI Ready")