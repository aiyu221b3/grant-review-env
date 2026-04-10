"""
Easy Grader
===========
Evaluates agent performance on task_easy.json.

Correct answer: APPROVE with high confidence.
No hidden flaws. Agent should approve after reading abstract
and optionally confirming methodology and budget.

Scoring:
- Correct decision (approve): +0.6
- Requested methodology before deciding: +0.1
- Requested budget before deciding: +0.1
- Confidence >= 0.7 on correct decision: +0.1
- Finished in <= 5 actions: +0.1 efficiency bonus
"""

from typing import Dict, Any

def normalize_to_strict_range(score: float) -> float:
    """
    Nudges scores of 0.0 to 0.01 and 1.0 to 0.99.
    Ensures compliance with strictly (0, 1) requirements.
    """
    epsilon = 0.01
    return max(epsilon, min(score, 1.0 - epsilon))
    
def grade_easy(episode_state: Dict[str, Any]) -> float:
    """
    Deterministic grader for easy task.
    
    Args:
        episode_state: dict from env.state() at episode end
        
    Returns:
        float score in [0.0, 1.0]
    """
    score = 0.0

    decision = episode_state.get("decision_made")
    sections = episode_state.get("sections_unlocked", [])
    step_count = episode_state.get("step_count", 0)
    cumulative_reward = episode_state.get("cumulative_reward", 0.0)

    # Correct decision
    if decision == "approve":
        score += 0.6
    elif decision is None:
        # Ran out of actions without deciding — partial credit if reward positive
        if cumulative_reward > 0:
            score += 0.1

    # Requested methodology
    if "methodology" in sections:
        score += 0.1

    # Requested budget
    if "budget" in sections:
        score += 0.1

    # Efficiency — finished in 5 or fewer actions
    if step_count <= 5:
        score += 0.1

    # Confidence bonus — can't directly access from state,
    # so we approximate: if decided quickly AND correctly, assume confident
    if decision == "approve" and step_count <= 4:
        score += 0.1
        
    # Calculate raw score, bound to [0, 1], then normalize to strictly (0, 1)
    raw_score = round(min(max(score, 0.0), 1.0), 4)
    return normalize_to_strict_range(raw_score)
