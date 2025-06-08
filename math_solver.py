import os
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
import google.generativeai as genai
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from tavily import TavilyClient
import dspy
import json
from langgraph.graph import StateGraph
from typing import TypedDict, Optional

load_dotenv()

with open("jeebench_math_results.json", "r", encoding="utf-8") as f:
    jee_results_cache = json.load(f)

jee_answer_lookup = {
    q["question"].strip(): q for q in jee_results_cache
}
# --- Input/Output Guardrails ---

def is_math_educational(text):
    keywords = [
        "math", "algebra", "geometry", "calculus", "integral", "derivative", "equation",
        "triangle", "area", "volume", "probability", "sum", "difference", "product",
        "root", "matrix", "vector", "limit", "function", "graph", "angle", "theorem",
        "mean", "median", "mode", "variance", "standard deviation", "logarithm",
        "exponent", "radius", "diameter", "circumference", "perimeter", "hypotenuse",
        "arithmetic", "number", "prime", "fraction", "decimal", "percent", "ratio",
        "proportion", "sequence", "series", "set"
    ]
    if re.search(r'[\d\+\-\*/\^=<>]', text):
        return True
    for kw in keywords:
        if kw in text.lower():
            return True
    return False

def filter_math_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s for s in sentences if is_math_educational(s)]

def rerank_chunks(chunks, question, top_k=3):
    question_words = set(re.findall(r'\w+', question.lower()))
    def score(chunk):
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        return len(question_words & chunk_words)
    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:top_k]

# Gemini API setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Qdrant setup
QDRANT_URL = os.getenv("QDRANT_URL", "https://1a3f0e99-06cc-40f1-a2ab-9c47668a0504.us-west-2-0.aws.cloud.qdrant.io")
COLLECTION_NAME = "math_kb"
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant = Qdrant(
    client=QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY")),
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
    content_payload_key="content",
)

# Load Gemini model
model = genai.GenerativeModel('models/gemini-1.5-flash')

# DSPy LLM wrapper
class GeminiModule(dspy.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text.strip()

gemini_module = GeminiModule(model)

# --- LangGraph Nodes ---

def input_guardrail_node(data):
    question = data["question"]
    print(f"🟩 Received question: {question}")
    if not is_math_educational(question):
        data["output"] = "Sorry, this app only answers math and educational questions."
        data["end"] = True
    return data

def kb_retrieval_node(data):
    question = data["question"]
    print("🔍 Searching KB...")
    docs = qdrant.similarity_search(question, k=3)
    docs = [doc for doc in docs if doc.page_content is not None]
    if docs:
        data["retrieved_context"] = "\n".join([doc.page_content for doc in docs])
        data["from_kb"] = True
    else:
        data["from_kb"] = False
        print("❌ No KB data found.")
    return data

def web_search_node(data):
    question = data["question"]
    print("🌐 No KB data. Searching Web...")
    web_result = tavily.search(question)
    if web_result and web_result['results']:
        raw_content = web_result['results'][0]['content']
        filtered_sentences = filter_math_sentences(raw_content)
        top_chunks = rerank_chunks(filtered_sentences, question, top_k=3)
        data["retrieved_context"] = " ".join(top_chunks) if top_chunks else "No relevant information found online."
        print("✅ Found in Web. Data:\n", data["retrieved_context"])
    else:
        data["retrieved_context"] = "No relevant information found online."
        print("❌ No relevant Web results.")
    return data

def llm_node(data):
    question = data["question"]
    retrieved_context = data.get("retrieved_context", "")
    prompt = f"""
You are a math professor. Answer the following math question using only the information provided below. 
Do not include information about other math topics. 
Be clear and concise, and explain step-by-step if the question requires it.

Question: {question}

Relevant information:
{retrieved_context}

Only answer the question above. Do not add unrelated math facts. If the question is not math-related, reply: 'Sorry, I can only answer math questions.'
"""
    data["output"] = gemini_module(prompt)
    print("🤖 Generating answer...")
    return data

def output_guardrail_node(data):
    output = data.get("output", "")
    if not is_math_educational(output):
        data["output"] = "Sorry, the answer could not be verified as math-focused. Please ask a math question."
    return data

# --- LangGraph Setup ---
from typing import TypedDict, Optional

class GraphState(TypedDict):
    question: str
    output: Optional[str]
    retrieved_context: Optional[str]
    from_kb: Optional[bool]
    end: Optional[bool]

graph = StateGraph(GraphState)

graph.add_node("input_guardrail", input_guardrail_node)
graph.add_node("kb_retrieval", kb_retrieval_node)
graph.add_node("web_search", web_search_node)
graph.add_node("llm", llm_node)
graph.add_node("output_guardrail", output_guardrail_node)

# Routing logic for KB retrieval
def kb_route(data):
    if data.get("from_kb"):
        return "llm"
    else:
        return "web_search"

graph.add_edge("input_guardrail", "kb_retrieval")
graph.add_conditional_edges("kb_retrieval", kb_route)
graph.add_edge("web_search", "llm")
graph.add_edge("llm", "output_guardrail")

# --- Agentic Answer Function using LangGraph ---
graph.set_entry_point("input_guardrail")  # Optional but good practice

runnable_graph = graph.compile()

def agentic_answer(question):
    data = {"question": question}
    result = runnable_graph.invoke(data)
    return result.get("output", "")

# --- Interactive code (only runs if executed directly) ---
if __name__ == "__main__":
    question = input("Ask a math question: ")

    answer = agentic_answer(question)
    print("\n🧠 Step-by-step solution:\n")
    print(answer)

    # --- Human-in-the-Loop Feedback ---
    if is_math_educational(answer):
        feedback = input("\nWas this answer helpful? (yes/no/needs improvement): ").strip().lower()
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Question: {question}\nAnswer: {answer}\nFeedback: {feedback}\n---\n")
        if feedback == "needs improvement":
            user_correction = input("Please describe what needs to be improved or add your correction: ")
            improved_prompt = f"""
The previous answer was: {answer}
The user said it needs improvement: "{user_correction}"
Please provide a revised, clearer, and more accurate step-by-step solution for the following question:

Question: {question}

Relevant information:
{answer}
"""
            improved_output = gemini_module(improved_prompt)
            print("\n🔄 Improved answer:\n")
            print(improved_output)
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"User Correction: {user_correction}\nImproved Answer: {improved_output}\n===\n")
            print("Thank you for your feedback! We've generated an improved answer.")
    else:
        print("Sorry, the answer could not be verified as math-focused. Please ask a math question.")