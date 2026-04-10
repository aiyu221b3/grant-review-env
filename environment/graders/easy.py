"""
Easy Grader
=============
Evaluates agent performance on task_easy.json.
Correct answer: APPROVE.
"""

from typing import Dict, Any

def normalize_to_strict_range(score: float) -> float:
    """Ensures score is strictly in (0, 1) to pass Phase 2 boundary checks."""
    epsilon = 0.01
    return max(epsilon, min(score, 1.0 - epsilon))

def grade_easy(episode_state: Dict[str, Any]) -> float:
    """Deterministic grader for easy task."""
    score = 0.0
    decision = episode_state.get("decision_made")
    sections = episode_state.get("sections_unlocked", [])
    step_count = episode_state.get("step_count", 0)

    # Reward for doing due diligence (reading sections)
    if "methodology" in sections or "budget" in sections or "team" in sections:
        score += 0.2

    # Core logic: Approve is correct
    if decision == "approve":
        score += 0.6
        # Efficiency bonus
        if step_count <= 4:
            score += 0.2
    elif decision == "reject":
        score -= 0.4

    # Bound raw score and apply strict normalization
    raw_score = round(min(max(score, 0.0), 1.0), 4)
    return normalize_to_strict_range(raw_score)
