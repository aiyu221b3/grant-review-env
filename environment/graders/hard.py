"""
Hard Grader
=============
Evaluates agent performance on task_hard.json.
Correct answer: REJECT.
Hidden flaw: Conflict of Interest (requires checking both Budget and Team).
"""

from typing import Dict, Any

def normalize_to_strict_range(score: float) -> float:
    """Ensures score is strictly in (0, 1) to pass Phase 2 boundary checks."""
    epsilon = 0.01
    return max(epsilon, min(score, 1.0 - epsilon))

def grade_hard(episode_state: Dict[str, Any]) -> float:
    """Deterministic grader for hard task."""
    score = 0.0
    decision = episode_state.get("decision_made")
    sections = episode_state.get("sections_unlocked", [])
    step_count = episode_state.get("step_count", 0)

    has_budget = "budget" in sections
    has_team = "team" in sections

    # Reward for investigating the right areas
    if has_budget:
        score += 0.15
    if has_team:
        score += 0.15
    
    # Bonus for putting the pieces together (cross-referencing)
    if has_budget and has_team:
        score += 0.1

    # Core logic: Reject is correct due to conflict
    if decision == "reject":
        score += 0.5
        # Generous efficiency bonus for hard task
        if step_count <= 6:
            score += 0.1
    elif decision == "approve":
        score -= 0.3
    elif decision is None and (has_budget or has_team):
        score += 0.05

    # Bound raw score and apply strict normalization
    raw_score = round(min(max(score, 0.0), 1.0), 4)
    return normalize_to_strict_range(raw_score)
