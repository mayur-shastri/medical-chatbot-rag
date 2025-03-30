import pandas as pd

# Load CSV file
csv_path = "C:\\Users\\Anuj\\Desktop\\Sem 6 DA\\Machine Learning J component\\ML Project Final cloned\\medquad_evaluation_results (5).csv"

df = pd.read_csv(csv_path)

# Calculate average scores
average_scores = {
    "Avg ROUGE-1": df["ROUGE-1"].mean(),
    "Avg ROUGE-L": df["ROUGE-L"].mean(),
    "Avg BLEU": df["BLEU"].mean(),
    "Avg F1 Score": df["F1"].mean(),
    "Avg Cosine Similarity": df["Cosine Similarity"].mean(),
}

# Identify best and worst performing queries
best_query = df.loc[df["F1"].idxmax()]
worst_query = df.loc[df["F1"].idxmin()]

# Print summary
print("==== Evaluation Summary ====")
for metric, score in average_scores.items():
    print(f"{metric}: {score:.4f}")

print("\n==== Best Performing Query ====")
print(f"Query: {best_query['Query']}")
print(f"Expected: {best_query['Expected Answer']}")
print(f"Generated: {best_query['Generated Answer']}")
print(f"F1 Score: {best_query['F1']:.4f}")

print("\n==== Worst Performing Query ====")
print(f"Query: {worst_query['Query']}")
print(f"Expected: {worst_query['Expected Answer']}")
print(f"Generated: {worst_query['Generated Answer']}")
print(f"F1 Score: {worst_query['F1']:.4f}")
