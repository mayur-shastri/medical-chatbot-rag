import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to FAISS vectorstore
DB_FAISS_PATH = "../vectorstore/db_faiss"

# HuggingFace model config
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Cache vectorstore
_vectorstore = None
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return _vectorstore

def set_custom_prompt():
    template = """
You are a diagnostic assistant. The user provides current symptoms and demographic data.
You are given context about:
- how symptoms map to possible diseases in different demographic groups
- how diseases manifest in different demographics with various symptoms

Your task is to:
1. Deduce the most likely causes based on symptom-to-disease evidence and demographic data
2. Cross-check those causes with how the diseases typically present
3. Eliminate unlikely possibilities
4. Return your reasoning in plain, readable English.

Only make conclusions based on the context provided.

User Profile:
{user_context}

Reference Documents:
{doc_context}

Question:
{question}

Begin your answer concisely but informatively.
"""
    return PromptTemplate(template=template, input_variables=["user_context", "doc_context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def get_rag_response(user_query, user_context):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Retrieve documents
    docs = retriever.get_relevant_documents(user_query)
    doc_context = " ".join([doc.page_content for doc in docs])

    # Set up QA chain with custom prompt
    chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )

    # Invoke the chain
    result = chain.invoke({
        "question": user_query,
        "user_context": user_context,
        "doc_context": doc_context
    })

    return result["result"]