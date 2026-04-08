"""
Hard Grader
===========
Evaluates agent performance on task_hard.json.

Correct answer: REJECT.
Hidden flaw: PI holds 40% equity in NeuralDrone Pvt. Ltd.
Budget includes $24,000 technology transfer fee to NeuralDrone.
Conflict only detectable by requesting BOTH budget AND team, then cross-referencing.

Agent must:
1. Request budget (sees NeuralDrone fee)
2. Request team (sees PI's equity stake)
3. Cross-reference to detect conflict
4. Reject the proposal

Scoring:
- Requested budget section: +0.15
- Requested team section: +0.15
- Both requested (cross-reference possible): +0.10 bonus
- Conflict detected in flaws_detected: +0.20
- Correct decision (reject): +0.40
- Wrong decision (approve): -0.30 penalty
- Asked clarification about conflict/NeuralDrone/equity: +0.10
"""

from typing import Dict, Any


def grade_hard(episode_state: Dict[str, Any]) -> float:
    """
    Deterministic grader for hard task.

    Args:
        episode_state: dict from env.state() at episode end

    Returns:
        float score in [0.0, 1.0]
    """
    score = 0.0

    decision = episode_state.get("decision_made")
    sections = episode_state.get("sections_unlocked", [])
    flaws_detected = episode_state.get("flaws_detected", [])

    budget_requested = "budget" in sections
    team_requested = "team" in sections

    # Requested budget — NeuralDrone fee visible here
    if budget_requested:
        score += 0.15

    # Requested team — PI equity visible here
    if team_requested:
        score += 0.15

    # Both requested — cross-reference possible
    if budget_requested and team_requested:
        score += 0.10

    # Conflict detected
    if "conflict_of_interest" in flaws_detected:
        score += 0.20

    # Correct decision
    if decision == "reject":
        score += 0.40
    elif decision == "approve":
        score -= 0.30
    elif decision is None:
        # No decision — partial credit if conflict was found
        if "conflict_of_interest" in flaws_detected:
            score += 0.10

    # Clarification about conflict area
    # Approximated: if conflict detected AND clarification was used
    # (conflict_of_interest appears in flaws when clarification probes conflict keywords)
    if "conflict_of_interest" in flaws_detected and budget_requested and team_requested:
        score += 0.10

    return round(min(max(score, 0.0), 1.0), 4)
