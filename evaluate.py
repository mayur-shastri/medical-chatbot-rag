# To Do : 
# 1. Evaluate metrics by swapping out the pre trained models, and comparing the metrics
# 2. LLM performance with and without the knowledge base's support - comparison
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv, find_dotenv

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the information provided in the context to answer the user's question as accurately and comprehensively as possible.
If the context does not contain sufficient information, say that you don't know—do not generate an answer outside the given context.

- Provide a well-structured and informative response.
- Include explanations, possible causes, symptoms, treatments, or precautions, only if the provided context has any, and if relevant.
- If the question involves a medical condition, include potential next steps a person might consider, such as consulting a specialist.

Context: {context}  
Question: {question}  

Begin your answer concisely but provide enough detail for clarity.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_answer(query, expected_answer, generated_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(generated_answer, expected_answer)
    bleu = sentence_bleu([expected_answer.split()], generated_answer.split())

    pred_tokens = generated_answer.lower().split()
    gt_tokens = expected_answer.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    emb_expected = embedder.embed_query(expected_answer)
    emb_generated = embedder.embed_query(generated_answer)
    emb_similarity = cosine_similarity(emb_expected, emb_generated)

    return {
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "BLEU": bleu,
        "F1": f1,
        "Cosine Similarity": emb_similarity
    }

def generate_answer_from_rag(query):
    response=qa_chain.invoke({'query': query})
    return response["result"]

# Load the MedQuad dataset
# dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train").select(range(50))

results = []

# Process each query
for item in dataset:
    query = item['Question']
    expected_answer = item['Answer']

    generated_answer = generate_answer_from_rag(query)

    metrics = evaluate_answer(query, expected_answer, generated_answer)
    metrics.update({
        "Query": query,
        "Expected Answer": expected_answer,
        "Generated Answer": generated_answer
    })
    results.append(metrics)

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("medquad_evaluation_results.csv", index=False, mode="a", header=False)
print("Evaluation completed! Results saved to medquad_evaluation_results.csv")