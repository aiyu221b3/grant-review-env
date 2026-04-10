"""
Medium Grader
=============
Evaluates agent performance on task_medium.json.

Correct answer: REJECT.
Hidden flaw: methodology labeling gap — no ground truth between monthly PHQ-9 administrations.

Agent must:
1. Request methodology section (flaw only visible there)
2. Detect the labeling gap (via clarification or just reading carefully)
3. Reject the proposal

Scoring:
- Correct decision (reject): +0.5
- Requested methodology section: +0.2 (flaw only findable here)
- Asked clarification about labels/training/validation: +0.15
- Detected flaw in flaws_detected list: +0.15 (via clarification probe)
- Wrong decision (approve): -0.2 penalty applied after base
"""

from typing import Dict, Any

def normalize_to_strict_range(score: float) -> float:
    """
    Nudges scores of 0.0 to 0.01 and 1.0 to 0.99.
    Ensures compliance with strictly (0, 1) requirements.
    """
    epsilon = 0.01
    return max(epsilon, min(score, 1.0 - epsilon))


def grade_medium(episode_state: Dict[str, Any]) -> float:
    """
    Deterministic grader for medium task.

    Args:
        episode_state: dict from env.state() at episode end

    Returns:
        float score in [0.0, 1.0]
    """
    score = 0.0

    decision = episode_state.get("decision_made")
    sections = episode_state.get("sections_unlocked", [])
    flaws_detected = episode_state.get("flaws_detected", [])
    step_count = episode_state.get("step_count", 0)

    # Requested methodology — necessary to find the flaw
    if "methodology" in sections:
        score += 0.2

    # Probed flaw area via clarification
    # flaws_detected gets populated when agent asks about methodology flaw keywords
    flaw_keywords = ["methodology_labeling_gap", "conflict_of_interest"]
    if any(f in flaw_keywords for f in flaws_detected):
        score += 0.15

    # Correct decision
    if decision == "reject":
        score += 0.5
    elif decision == "approve":
        # Wrong decision — penalize
        score -= 0.2
    elif decision is None:
        # No decision made — partial credit only if methodology was read
        if "methodology" in sections:
            score += 0.05

    # Efficiency bonus — but only if correct
    if decision == "reject" and step_count <= 4:
        score += 0.15

 return normalize_to_strict_range(final_score)
