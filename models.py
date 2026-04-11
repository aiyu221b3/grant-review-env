"""
models.py
Pydantic typed models for the Grant Review Environment.
Defines Observation, Action, and Reward structures per OpenEnv spec.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# enums for strict typing so the llm doesn't make up random actions
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


# observation model. this is what the agent actually sees. 
# we hide the full proposal at first.
class GrantProposalObservation(BaseModel):
    # agent always gets to see these
    abstract: str = Field(..., description="Proposal abstract — always visible")
    title: str = Field(..., description="Proposal title")
    requested_amount: float = Field(..., description="Funding amount requested in USD")

    # these start empty. agent has to use actions to unlock them.
    methodology: Optional[str] = Field(None, description="Methodology section — unlocked by REQUEST_METHODOLOGY")
    budget_breakdown: Optional[Dict[str, float]] = Field(None, description="Budget breakdown — unlocked by REQUEST_BUDGET")
    team_composition: Optional[List[str]] = Field(None, description="Team members and roles — unlocked by REQUEST_TEAM")
    references: Optional[List[str]] = Field(None, description="References — unlocked by REQUEST_REFERENCES")

    # if agent asks something, the applicant's reply goes here
    clarification_response: Optional[str] = Field(None, description="Applicant response to last clarification")

    # tracking stuff so agent knows how much time it has left
    actions_remaining: int = Field(..., description="Remaining actions before forced decision")
    sections_unlocked: List[str] = Field(default_factory=list, description="Which sections agent has unlocked")
    step_number: int = Field(..., description="Current step in episode")
    last_action_error: Optional[str] = Field(None, description="Error from last action if any")

    # giving the agent the rubric so it knows what we want it to grade on
    evaluation_criteria: Dict[str, float] = Field(
        default_factory=dict,
        description="Rubric criteria and their weights"
    )


# action model. what the agent can do. optionals because it doesn't 
# always need to justify simply asking for the budget.
class GrantReviewAction(BaseModel):
    action_type: ActionType = Field(..., description="Type of action to take")

    # only used when asking a direct question
    clarification_question: Optional[str] = Field(
        None,
        description="Specific question to ask applicant — only for REQUEST_CLARIFICATION"
    )

    # only needed when making the final call
    justification: Optional[str] = Field(
        None,
        description="Reasoning for funding decision — only for APPROVE or REJECT"
    )
    
    # keeping confidence bounded between 0 and 1 for easy math later
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Agent confidence in decision 0.0-1.0 — only for APPROVE or REJECT"
    )


# reward model. breaking down the score so we know exactly why 
# the agent got points instead of just giving a single number.
class GrantReviewReward(BaseModel):
    # the actual points given this turn
    step_reward: float = Field(..., description="Reward for this specific step")

    # breaking it down for debugging
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

    # tracking total score so far
    cumulative_reward: float = Field(..., description="Total reward accumulated so far")


# standard step result for openenv.
class StepResult(BaseModel):
    observation: GrantProposalObservation
    reward: float
    done: bool
    info: Dict = Field(default_factory=dict)
