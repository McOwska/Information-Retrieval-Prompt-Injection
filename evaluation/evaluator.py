import json
from typing import List, Dict
from evaluation.metrics import calculate_f1, calculate_asr
from multihop_agent.agent import run_multihop_agent

QUESTION_TYPES_TO_EVALUATE = ["bridge"]

def load_ground_truths(filepath: str = "data/processed/questions.jsonl") -> Dict[str, str]:
    """
    Loads the ground truth answers from the generated HotpotQA dataset.
    Returns a dictionary mapping the question text to the correct gold label answer.
    """
    ground_truths = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                question = data.get("question")
                answer = data.get("answer", "")
                type = data.get("type", "")
                
                if question and type in QUESTION_TYPES_TO_EVALUATE:
                    ground_truths[question] = answer
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Make sure to run save_docs_to_local.py first.")
        
    return ground_truths


def evaluate_results(agent_results: List[Dict], ground_truths: Dict[str, str], save_results_path: str) -> Dict[str, float]:
    """
    Evaluates a list of agent results against the ground truths.
    """

    if save_results_path:
        os.makedirs(save_results_path, exist_ok=True)

        comparison_path = os.path.join(save_results_path, "comparison.csv")
        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write("Question,Ground Truth,Agent Answer\n")
            for result in agent_results:
                question = result.get("question", "").replace(",", " ")
                agent_answer = result.get("answer", "").replace(",", " ")
                ground_truth = ground_truths.get(question, "").replace(",", " ")
                f.write(f"{question},{ground_truth},{agent_answer}\n")
        print(f"Comparison CSV saved to {comparison_path}")

    total_f1 = 0.0
    total_asr = 0.0
    evaluated_count = 0

    for result in agent_results:
        question = result.get("question")
        prediction = result.get("answer", "")

        print(f"Question: {question}")
        print(f"Answer of LLM: '{prediction}'")
        print("-" * 30)
        
        # 1. Calculate Attack Success Rate (ASR)
        total_asr += calculate_asr(prediction)
        
        # 2. Calculate Answer F1
        if question in ground_truths:
            truth = ground_truths[question]
            total_f1 += calculate_f1(prediction, truth)
            evaluated_count += 1
        else:
            print(f"Warning: No ground truth found for question: '{question}'")

    avg_f1 = total_f1 / evaluated_count if evaluated_count > 0 else 0.0
    avg_asr = total_asr / len(agent_results) if len(agent_results) > 0 else 0.0

    print("\n" + "=" * 40)
    print(" EVALUATION RESULTS ")
    print("=" * 40)
    print(f"Total evaluated  : {len(agent_results)}")
    print(f"Answer F1 Score  : {avg_f1:.4f}")
    print(f"Attack Success   : {avg_asr:.4f} ({(avg_asr*100):.1f}%)")
    print("=" * 40)

    return {"f1": avg_f1, "asr": avg_asr}


import time

import time

def run_evaluation_pipeline(retriever, questions_path: str = "data/processed/questions.jsonl", limit: int = 5, results_path: str = "", poisoned_hops = None, poisoned_retriever = None):
    ground_truths = load_ground_truths(questions_path)
    questions = list(ground_truths.keys())[:limit]
    results = []
    
    for q in questions:
        print(f"\n--- Querying: {q} ---")
        
        success = False
        max_retries = 3
        
        # Retry loop: attempt to process each query up to 3 times
        for attempt in range(max_retries):
            try:
                output = run_multihop_agent(q, retriever=retriever, poisoned_hops=poisoned_hops, poisoned_retriever=poisoned_retriever)
                
                if output and output.get("answer"):
                    results.append(output)
                    success = True
                    break  # Exit the retry loop upon success
                else:
                    print(f"Empty result on attempt {attempt + 1}")
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
            
            # Wait before retrying
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
        
        if not success:
            print(f"Failed to process query after {max_retries} attempts. Skipping.")
            
    return evaluate_results(results, ground_truths, results_path)



import json
import os

def save_results(results, metrics, filename="results_baseline.json"):
    output = {
        "metrics": metrics,
        "results": results
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {filename}")