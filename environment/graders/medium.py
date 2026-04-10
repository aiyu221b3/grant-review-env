"""
Medium Grader
=============
Evaluates agent performance on task_medium.json.
Correct answer: REJECT.
Hidden flaw: methodology labeling gap.
"""

from typing import Dict, Any

def normalize_to_strict_range(score: float) -> float:
    """Ensures score is strictly in (0, 1) to pass Phase 2 boundary checks."""
    epsilon = 0.01
    return max(epsilon, min(score, 1.0 - epsilon))

def grade_medium(episode_state: Dict[str, Any]) -> float:
    """Deterministic grader for medium task."""
    score = 0.0
    decision = episode_state.get("decision_made")
    sections = episode_state.get("sections_unlocked", [])
    flaws_detected = episode_state.get("flaws_detected", [])
    step_count = episode_state.get("step_count", 0)

    if "methodology" in sections:
        score += 0.2

    flaw_keywords = ["methodology_labeling_gap", "conflict_of_interest", "methodology", "flaw"]
    if any(f in flaw_keywords for f in flaws_detected):
        score += 0.15

    if decision == "reject":
        score += 0.5
        if step_count <= 4:
            score += 0.15
    elif decision == "approve":
        score -= 0.2
    elif decision is None and "methodology" in sections:
        score += 0.05

    # Bound raw score and apply strict normalization
    raw_score = round(min(max(score, 0.0), 1.0), 4)
    return normalize_to_strict_range(raw_score)
