import json
from typing import List, Dict
from evaluation.metrics import calculate_f1, calculate_asr

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
                
                if question:
                    ground_truths[question] = answer
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Make sure to run save_docs_to_local.py first.")
        
    return ground_truths


def evaluate_results(agent_results: List[Dict], ground_truths: Dict[str, str]) -> Dict[str, float]:
    """
    Evaluates a list of agent results against the ground truths.
    
    Args:
        agent_results: List of dicts returned by run_multihop_agent().
                       Expected to have 'question' and 'answer' keys.
        ground_truths: Dict mapping the question text to the gold label answer.
        
    Returns:
        Dictionary containing the aggregated F1 and ASR scores.
    """
    total_f1 = 0.0
    total_asr = 0.0
    evaluated_count = 0

    for result in agent_results:
        question = result.get("question")
        prediction = result.get("answer", "")
        
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