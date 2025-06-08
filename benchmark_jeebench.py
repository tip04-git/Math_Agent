import json
from math_solver import gemini_module, is_math_educational, filter_math_sentences, rerank_chunks, qdrant, tavily

# Load your JEE Bench dataset (replace with your actual path)
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter for math questions
math_questions = [q for q in data if q.get("subject") == "math" and "question" in q and "gold" in q]

def agentic_answer(question):
    # This is a simplified version of your math_solver pipeline
    if not is_math_educational(question):
        return "Sorry, this app only answers math and educational questions."
    docs = qdrant.similarity_search(question, k=3)
    docs = [doc for doc in docs if doc.page_content is not None]
    if not docs:
        web_result = tavily.search(question)
        if web_result and web_result['results']:
            raw_content = web_result['results'][0]['content']
            filtered_sentences = filter_math_sentences(raw_content)
            top_chunks = rerank_chunks(filtered_sentences, question, top_k=3)
            retrieved_context = " ".join(top_chunks) if top_chunks else "No relevant information found online."
        else:
            retrieved_context = "No relevant information found online."
    else:
        retrieved_context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are a math professor. Answer the following math question using only the information provided below. 
Do not include information about other math topics. 
Be clear and concise, and explain step-by-step if the question requires it.

Question: {question}

Relevant information:
{retrieved_context}

Only answer the question above. Do not add unrelated math facts. If the question is not math-related, reply: 'Sorry, I can only answer math questions.'
"""
    return gemini_module(prompt)

def match_answer(model_answer, gold):
    # Simple matching: check if gold option/number is in model answer (case-insensitive)
    gold = str(gold).strip().upper()
    model_answer = str(model_answer).strip().upper()
    # For MCQ: check if any gold letter is in model answer
    if gold in model_answer:
        return True
    # For multi-answer: all letters must be present
    if all(opt in model_answer for opt in gold):
        return True
    return False

results = []
correct = 0
total = 0

import time
for q in math_questions:
    question = q["question"]
    gold = q["gold"]
    model_ans = agentic_answer(question)
    is_correct = match_answer(model_ans, gold)
    results.append({
        "question": question,
        "gold": gold,
        "model_answer": model_ans,
        "correct": is_correct
    })
    total += 1
    if is_correct:
        correct += 1
    print(f"Q: {question}\nGold: {gold}\nModel: {model_ans}\nCorrect: {is_correct}\n---")
    time.sleep(5)  # To avoid hitting API rate limits

print(f"\nMath Benchmark Accuracy: {correct}/{total} = {correct/total:.2%}")

# Save results for further analysis
with open("jeebench_math_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)