"""
models.py
Pydantic typed models for the Grant Review Environment.
Defines Observation, Action, and Reward structures per OpenEnv spec.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    REQUEST_METHODOLOGY = "request_methodology"
    REQUEST_BUDGET = "request_budget"
    REQUEST_TEAM = "request_team"
    REQUEST_REFERENCES = "request_references"
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CLARIFICATION = "request_clarification"


class ProposalSection(str, Enum):
    ABSTRACT = "abstract"
    METHODOLOGY = "methodology"
    BUDGET = "budget"
    TEAM = "team"
    REFERENCES = "references"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Observation Model
# What the agent sees at each step
# ---------------------------------------------------------------------------

class GrantProposalObservation(BaseModel):
    """
    Partial observation of the grant proposal.
    Agent starts with abstract only.
    Each action unlocks more sections.
    """
    # Always visible
    abstract: str = Field(..., description="Proposal abstract — always visible")
    title: str = Field(..., description="Proposal title")
    requested_amount: float = Field(..., description="Funding amount requested in USD")

    # Unlocked progressively via actions
    methodology: Optional[str] = Field(None, description="Methodology section — unlocked by REQUEST_METHODOLOGY")
    budget_breakdown: Optional[Dict[str, float]] = Field(None, description="Budget breakdown — unlocked by REQUEST_BUDGET")
    team_composition: Optional[List[str]] = Field(None, description="Team members and roles — unlocked by REQUEST_TEAM")
    references: Optional[List[str]] = Field(None, description="References — unlocked by REQUEST_REFERENCES")

    # Applicant response to last clarification request
    clarification_response: Optional[str] = Field(None, description="Applicant response to last clarification")

    # State tracking
    actions_remaining: int = Field(..., description="Remaining actions before forced decision")
    sections_unlocked: List[str] = Field(default_factory=list, description="Which sections agent has unlocked")
    step_number: int = Field(..., description="Current step in episode")
    last_action_error: Optional[str] = Field(None, description="Error from last action if any")

    # Rubric hints (always visible — agent knows what it's evaluating against)
    evaluation_criteria: Dict[str, float] = Field(
        default_factory=dict,
        description="Rubric criteria and their weights"
    )


# ---------------------------------------------------------------------------
# Action Model
# What the agent can do
# ---------------------------------------------------------------------------

class GrantReviewAction(BaseModel):
    """
    Agent action in the grant review environment.
    Either request more information or make a final decision.
    """
    action_type: ActionType = Field(..., description="Type of action to take")

    # Only used when action_type is REQUEST_CLARIFICATION
    clarification_question: Optional[str] = Field(
        None,
        description="Specific question to ask applicant — only for REQUEST_CLARIFICATION"
    )

    # Only used when action_type is APPROVE or REJECT
    justification: Optional[str] = Field(
        None,
        description="Reasoning for funding decision — only for APPROVE or REJECT"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Agent confidence in decision 0.0-1.0 — only for APPROVE or REJECT"
    )


# ---------------------------------------------------------------------------
# Reward Model
# Partial progress signal throughout episode
# ---------------------------------------------------------------------------

class GrantReviewReward(BaseModel):
    """
    Reward signal per step.
    Not just binary end-of-episode — rewards partial progress.
    """
    # Step reward
    step_reward: float = Field(..., description="Reward for this specific step")

    # Breakdown for interpretability
    information_gain: float = Field(
        0.0,
        description="Reward for unlocking a section that contained relevant signal"
    )
    flaw_detection_bonus: float = Field(
        0.0,
        description="Bonus if agent detected a hidden flaw this step"
    )
    efficiency_penalty: float = Field(
        0.0,
        description="Penalty for requesting redundant or irrelevant information"
    )
    decision_quality: float = Field(
        0.0,
        description="Final reward for correct/incorrect funding decision — non-zero only at episode end"
    )

    # Running total
    cumulative_reward: float = Field(..., description="Total reward accumulated so far")


# ---------------------------------------------------------------------------
# Step Result
# What env.step() returns
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: GrantProposalObservation
    reward: float
    done: bool
    info: Dict = Field(default_factory=dict)
