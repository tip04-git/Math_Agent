import json
from difflib import SequenceMatcher
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Load results
with open("jeebench_math_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

def partial_credit(model_answer, gold):
    gold_set = set(gold.upper())
    model_set = set(model_answer.upper())
    if not gold_set:
        return 0
    return len(gold_set & model_set) / len(gold_set)

def extract_answer(text):
    # Try to extract only the answer part (e.g., last line with A/B/C/D or number)
    lines = text.strip().splitlines()
    for line in reversed(lines):
        # Look for a line that is just options or a number
        m = re.match(r"^\s*([A-D]+|\d+(\.\d+)?|[A-D])\s*$", line.strip().upper())
        if m:
            return m.group(1)
    # Fallback: return the whole text
    return text.strip().upper()

def fuzzy_match(a, b, threshold=0.8):
    a_ans = extract_answer(a)
    b_ans = extract_answer(b)
    return SequenceMatcher(None, a_ans, b_ans).ratio() >= threshold

exact_matches = 0
partial_scores = 0
fuzzy_matches = 0
total = len(results)
incorrect = []

# For confusion matrix (single-answer MCQ)
labels = ["A", "B", "C", "D"]
confusion = {g: Counter() for g in labels}

# For per-option confusion (multi-answer MCQ)
option_labels = ["A", "B", "C", "D"]
option_confusion = {opt: {"TP":0, "FP":0, "FN":0} for opt in option_labels}

for r in results:
    gold = r["gold"].strip().upper()
    model = r["model_answer"].strip().upper()
    # Exact match (already in your results)
    if r.get("correct"):
        exact_matches += 1
    # Partial credit (for multi-answer MCQ)
    partial = partial_credit(model, gold)
    partial_scores += partial
    # Fuzzy match (for numeric/text answers)
    if fuzzy_match(model, gold):
        fuzzy_matches += 1
    # Collect incorrect answers for error analysis
    if not r.get("correct"):
        incorrect.append({
            "question": r["question"],
            "gold": gold,
            "model_answer": model
        })
    # Confusion matrix: only for single-letter answers
    if len(gold) == 1 and len(model) == 1 and gold in labels and model in labels:
        confusion[gold][model] += 1
    # Per-option confusion for multi-answer MCQ
    gold_set = set(gold)
    model_set = set(model)
    for opt in option_labels:
        if opt in gold_set and opt in model_set:
            option_confusion[opt]["TP"] += 1  # True Positive
        elif opt not in gold_set and opt in model_set:
            option_confusion[opt]["FP"] += 1  # False Positive
        elif opt in gold_set and opt not in model_set:
            option_confusion[opt]["FN"] += 1  # False Negative

single_mcq_count = sum(
    1 for r in results
    if len(r["gold"].strip().upper()) == 1 and len(r["model_answer"].strip().upper()) == 1
    and r["gold"].strip().upper() in labels and r["model_answer"].strip().upper() in labels
)
print(f"Single-answer MCQ cases: {single_mcq_count}")

print(f"Total questions: {total}")
print(f"Exact match accuracy: {exact_matches}/{total} = {exact_matches/total:.2%}")
print(f"Average partial credit: {partial_scores/total:.2%}")
print(f"Fuzzy match accuracy: {fuzzy_matches}/{total} = {fuzzy_matches/total:.2%}")
print(f"Incorrect answers: {len(incorrect)}")

# Print confusion matrix
print("\nConfusion Matrix (single-answer MCQ):")
header = "     " + "  ".join(labels)
print(header)
for g in labels:
    row = f"{g}: " + "  ".join(str(confusion[g][m]) for m in labels)
    print(row)

# Print per-option confusion for multi-answer MCQ
print("\nPer-option confusion (multi-answer MCQ):")
for opt in option_labels:
    print(f"{opt}: TP={option_confusion[opt]['TP']} FP={option_confusion[opt]['FP']} FN={option_confusion[opt]['FN']}")

# Save incorrect answers for manual review
with open("incorrect_answers.json", "w", encoding="utf-8") as f:
    json.dump(incorrect, f, indent=2)

print("Incorrect answers saved to incorrect_answers.json")

# --- Visualize Confusion Matrix ---
conf_matrix = np.array([[confusion[g][m] for m in labels] for g in labels])

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Model Answer")
plt.ylabel("Gold Answer")
plt.title("Confusion Matrix (Single-answer MCQ)")
plt.tight_layout()
plt.show()

# --- Visualize Metrics ---
metrics = {
    "Exact Match Accuracy": exact_matches/total,
    "Average Partial Credit": partial_scores/total,
    "Fuzzy Match Accuracy": fuzzy_matches/total
}

plt.figure(figsize=(7,4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Model Performance Metrics")
plt.tight_layout()
plt.show()

# --- Visualize Correct vs Incorrect as Pie Chart ---
correct_count = exact_matches
incorrect_count = total - exact_matches

plt.figure(figsize=(5,5))
plt.pie(
    [correct_count, incorrect_count],
    labels=["Correct", "Incorrect"],
    autopct='%1.1f%%',
    colors=["#4CAF50", "#F44336"],
    startangle=90
)
plt.title("Correct vs Incorrect Answers")
plt.tight_layout()
plt.show()

# --- Visualize Per-option Confusion as Bar Chart ---
tp = [option_confusion[opt]["TP"] for opt in option_labels]
fp = [option_confusion[opt]["FP"] for opt in option_labels]
fn = [option_confusion[opt]["FN"] for opt in option_labels]

x = np.arange(len(option_labels))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, tp, width, label='True Positive', color='#4CAF50')
plt.bar(x, fp, width, label='False Positive', color='#FFC107')
plt.bar(x + width, fn, width, label='False Negative', color='#F44336')
plt.xticks(x, option_labels)
plt.ylabel("Count")
plt.title("Per-option Confusion (Multi-answer MCQ)")
plt.legend()
plt.tight_layout()
plt.show()