from langchain_huggingface import HuggingFaceEmbeddings
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# To Do : Evaluate metrics by swapping out the pre trained models, and comparing the metrics

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_answer(query, expected_answer, generated_answer):
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(generated_answer, expected_answer)
    
    # BLEU
    bleu = sentence_bleu([expected_answer.split()], generated_answer.split())
    
    # Token F1
    pred_tokens = generated_answer.lower().split()
    gt_tokens = expected_answer.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Embedding-based similarity
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

