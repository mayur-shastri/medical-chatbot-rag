import os
import re

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
Use the information provided in the context and user profile to answer the user's question as accurately and comprehensively as possible.
If the context does not contain sufficient information, say you don't know—do not fabricate information.

- If relevant, incorporate symptoms, causes, or treatments mentioned in the context.
- If the question involves a health concern, give possible next steps (e.g., consult a specialist).

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

User Profile & Chat Context:
{context}

Question:
{question}

Begin your answer concisely but informatively.
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def highlight_terms(text, context):
    """Highlight words from the context that appear in the chatbot response by making them bold."""
    terms = set(re.findall(r'\b[A-Za-z-]+\b', context))
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
        text = pattern.sub(f'**{term}**', text)
    return text

def get_rag_response(user_query, user_context):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()}
    )

    result = chain.invoke({
        "question": user_query,
        "context": user_context
    })

    if not result:
        return "No response generated."

    answer = result["result"]
    context_text = " ".join([doc.page_content for doc in result["source_documents"]])
    highlighted = highlight_terms(answer, context_text)
    return highlighted
