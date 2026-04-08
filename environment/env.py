"""
env.py
Grant Proposal Review Environment — OpenEnv compliant.
Implements step() / reset() / state() API with typed Pydantic models.

An AI agent learns to evaluate research grant proposals like a real
funding committee — requesting information strategically, detecting
hidden weaknesses, and making funding decisions under uncertainty.
"""

import json
import os
from openenv_core.env_server import Environment
from pathlib import Path
from typing import Dict, Optional

from .applicant import ApplicantProfile, StrategicApplicant
from .models import (
    ActionType,
    GrantProposalObservation,
    GrantReviewAction,
    GrantReviewReward,
    StepResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ACTIONS = 8          # Agent's action budget per episode
TASKS_DIR = Path(__file__).parent.parent / "tasks"


# ---------------------------------------------------------------------------
# Reward shaping constants
# Partial progress signal — not just binary end reward
# ---------------------------------------------------------------------------

REWARD_RELEVANT_INFO = 0.15       # Unlocked a section with signal
REWARD_IRRELEVANT_INFO = -0.05    # Wasted action on low-value section
REWARD_FLAW_DETECTED = 0.25       # Agent detected a hidden flaw via clarification
REWARD_CORRECT_DECISION = 1.0     # Correct funding decision
REWARD_WRONG_DECISION = -0.5      # Wrong funding decision
REWARD_CORRECT_WITH_CONFIDENCE = 0.2   # Bonus for high confidence + correct
REWARD_FORCED_DECISION_PENALTY = -0.1  # Ran out of actions, forced decision


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GrantReviewEnv:
    """
    OpenEnv-compliant Grant Proposal Review Environment.

    Observation space: Partial view of grant proposal.
                      Agent starts with abstract only.
                      Actions unlock additional sections.

    Action space: Discrete.
                 REQUEST_METHODOLOGY, REQUEST_BUDGET, REQUEST_TEAM,
                 REQUEST_REFERENCES, REQUEST_CLARIFICATION,
                 APPROVE, REJECT

    Reward: Dense partial progress signal throughout episode.
            Final reward based on decision correctness.

    Episode ends when: agent makes APPROVE/REJECT decision,
                       or action budget is exhausted.
    """
    
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self._task_data: Optional[Dict] = None
        self._applicant: Optional[StrategicApplicant] = None
        self._step_count: int = 0
        self._actions_remaining: int = MAX_ACTIONS
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._flaws_detected: list = []
        self._last_observation: Optional[GrantProposalObservation] = None
        self._last_error: Optional[str] = None
        self._decision_made: Optional[str] = None

        self._load_task(task_name)

    # -----------------------------------------------------------------------
    # OpenEnv required methods
    # -----------------------------------------------------------------------
    
    def reset(self) -> StepResult:
        """
        Reset environment to initial state.
        Returns initial observation — abstract only.
        """
        self._load_task(self.task_name)
        self._step_count = 0
        self._actions_remaining = MAX_ACTIONS
        self._done = False
        self._cumulative_reward = 0.0
        self._flaws_detected = []
        self._last_error = None
        self._decision_made = None

        obs = self._build_observation()
        self._last_observation = obs

        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={"message": "Episode started. You have access to the proposal abstract."}
        )

    def step(self, action: GrantReviewAction) -> StepResult:
        """
        Execute one action in the environment.
        Returns updated observation, reward, done flag, and info.
        """
        if self._done:
            return StepResult(
                observation=self._last_observation,
                reward=0.0,
                done=True,
                info={"message": "Episode already finished. Call reset()."}
            )

        self._step_count += 1
        self._actions_remaining -= 1
        self._last_error = None
        reward_breakdown = GrantReviewReward(
            step_reward=0.0,
            cumulative_reward=self._cumulative_reward
        )

        # --- Handle each action type ---

        if action.action_type == ActionType.REQUEST_METHODOLOGY:
            reward_breakdown = self._handle_request_methodology(reward_breakdown)

        elif action.action_type == ActionType.REQUEST_BUDGET:
            reward_breakdown = self._handle_request_budget(reward_breakdown)

        elif action.action_type == ActionType.REQUEST_TEAM:
            reward_breakdown = self._handle_request_team(reward_breakdown)

        elif action.action_type == ActionType.REQUEST_REFERENCES:
            reward_breakdown = self._handle_request_references(reward_breakdown)

        elif action.action_type == ActionType.REQUEST_CLARIFICATION:
            reward_breakdown = self._handle_clarification(
                action.clarification_question or "", reward_breakdown
            )

        elif action.action_type in (ActionType.APPROVE, ActionType.REJECT):
            reward_breakdown = self._handle_decision(action, reward_breakdown)
            self._done = True
            self._decision_made = action.action_type.value

        # --- Check if action budget exhausted ---
        if self._actions_remaining <= 0 and not self._done:
            reward_breakdown.step_reward += REWARD_FORCED_DECISION_PENALTY
            reward_breakdown.step_reward = round(reward_breakdown.step_reward, 4)
            self._done = True
            self._last_error = "Action budget exhausted. Episode ended without decision."

        # --- Update cumulative reward ---
        self._cumulative_reward += reward_breakdown.step_reward
        reward_breakdown.cumulative_reward = round(self._cumulative_reward, 4)

        obs = self._build_observation()
        self._last_observation = obs

        return StepResult(
            observation=obs,
            reward=round(reward_breakdown.step_reward, 4),
            done=self._done,
            info={
                "reward_breakdown": reward_breakdown.model_dump(),
                "flaws_detected": self._flaws_detected,
                "decision": self._decision_made,
                "error": self._last_error,
            }
        )
    @property
    def state(self) -> Dict:
        """
        Returns full current environment state.
        Used by OpenEnv spec — returns everything including hidden state.
        """
        return {
            "task_name": self.task_name,
            "step_count": self._step_count,
            "actions_remaining": self._actions_remaining,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "flaws_detected": self._flaws_detected,
            "decision_made": self._decision_made,
            "sections_unlocked": self._applicant.get_sections_revealed() if self._applicant else [],
            "ground_truth": {
                "should_be_funded": self._applicant.get_correct_decision() if self._applicant else None,
                "hidden_flaws": self._applicant.get_hidden_flaws() if self._applicant else [],
            }
        }

    def close(self):
        """Clean up environment resources."""
        pass

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_request_methodology(self, reward: GrantReviewReward) -> GrantReviewReward:
        if "methodology" in self._applicant.get_sections_revealed():
            reward.step_reward += REWARD_IRRELEVANT_INFO
            reward.efficiency_penalty = REWARD_IRRELEVANT_INFO
            self._last_error = "Methodology already requested."
        else:
            self._applicant.reveal_methodology()
            # Reward based on whether methodology had signal
            flaws = self._applicant.get_hidden_flaws()
            has_methodology_flaw = any("methodology" in f.lower() for f in flaws)
            if has_methodology_flaw:
                reward.step_reward += REWARD_RELEVANT_INFO
                reward.information_gain = REWARD_RELEVANT_INFO
            else:
                reward.step_reward += REWARD_IRRELEVANT_INFO * 0.5
        reward.step_reward = round(reward.step_reward, 4)
        return reward

    def _handle_request_budget(self, reward: GrantReviewReward) -> GrantReviewReward:
        if "budget" in self._applicant.get_sections_revealed():
            reward.step_reward += REWARD_IRRELEVANT_INFO
            reward.efficiency_penalty = REWARD_IRRELEVANT_INFO
            self._last_error = "Budget already requested."
        else:
            self._applicant.reveal_budget()
            flaws = self._applicant.get_hidden_flaws()
            has_budget_flaw = any("budget" in f.lower() or "conflict" in f.lower() for f in flaws)
            if has_budget_flaw:
                reward.step_reward += REWARD_RELEVANT_INFO
                reward.information_gain = REWARD_RELEVANT_INFO
            else:
                reward.step_reward += REWARD_IRRELEVANT_INFO * 0.5
        reward.step_reward = round(reward.step_reward, 4)
        return reward

    def _handle_request_team(self, reward: GrantReviewReward) -> GrantReviewReward:
        if "team" in self._applicant.get_sections_revealed():
            reward.step_reward += REWARD_IRRELEVANT_INFO
            reward.efficiency_penalty = REWARD_IRRELEVANT_INFO
            self._last_error = "Team composition already requested."
        else:
            self._applicant.reveal_team()
            # If both budget and team now revealed, conflict detectable
            if self._applicant.cross_reference_detectable():
                reward.step_reward += REWARD_FLAW_DETECTED
                reward.flaw_detection_bonus = REWARD_FLAW_DETECTED
                if "conflict_of_interest" not in self._flaws_detected:
                    self._flaws_detected.append("conflict_of_interest")
            else:
                reward.step_reward += REWARD_IRRELEVANT_INFO * 0.5
        reward.step_reward = round(reward.step_reward, 4)
        return reward

    def _handle_request_references(self, reward: GrantReviewReward) -> GrantReviewReward:
        if "references" in self._applicant.get_sections_revealed():
            reward.step_reward += REWARD_IRRELEVANT_INFO
            self._last_error = "References already requested."
        else:
            self._applicant.reveal_references()
            reward.step_reward += 0.0  # References rarely contain critical signal
        reward.step_reward = round(reward.step_reward, 4)
        return reward

    def _handle_clarification(self, question: str, reward: GrantReviewReward) -> GrantReviewReward:
        if not question:
            reward.step_reward += REWARD_IRRELEVANT_INFO
            self._last_error = "Clarification question was empty."
            reward.step_reward = round(reward.step_reward, 4)
            return reward

        response = self._applicant.respond_to_clarification(question)

        # Check if question probed a flaw area
        flaw_keywords = ["conflict", "interest", "methodology", "missing", "incomplete", "budget"]
        question_lower = question.lower()
        probed_flaw = any(kw in question_lower for kw in flaw_keywords)

        if probed_flaw:
            reward.step_reward += REWARD_RELEVANT_INFO * 0.5
            reward.information_gain = REWARD_RELEVANT_INFO * 0.5
        else:
            reward.step_reward += REWARD_IRRELEVANT_INFO * 0.3

        reward.step_reward = round(reward.step_reward, 4)
        return reward

    def _handle_decision(self, action: GrantReviewAction, reward: GrantReviewReward) -> GrantReviewReward:
        correct = self._applicant.get_correct_decision()
        approved = action.action_type == ActionType.APPROVE

        if approved == correct:
            reward.step_reward += REWARD_CORRECT_DECISION
            reward.decision_quality = REWARD_CORRECT_DECISION
            # Bonus for high confidence correct decision
            if action.confidence and action.confidence >= 0.8:
                reward.step_reward += REWARD_CORRECT_WITH_CONFIDENCE
                reward.decision_quality += REWARD_CORRECT_WITH_CONFIDENCE
            # Bonus for detecting flaws before deciding
            flaw_bonus = len(self._flaws_detected) * 0.1
            reward.step_reward += flaw_bonus
        else:
            reward.step_reward += REWARD_WRONG_DECISION
            reward.decision_quality = REWARD_WRONG_DECISION

        reward.step_reward = round(reward.step_reward, 4)
        return reward

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_task(self, task_name: str):
        """Load task JSON and initialize applicant."""
        task_file = TASKS_DIR / f"task_{task_name}.json"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file) as f:
            self._task_data = json.load(f)

        profile = ApplicantProfile(**self._task_data)
        self._applicant = StrategicApplicant(profile)

    def _build_observation(self) -> GrantProposalObservation:
        """Build current observation from applicant state."""
        revealed = self._applicant.get_sections_revealed()

        return GrantProposalObservation(
            abstract=self._applicant.get_abstract(),
            title=self._applicant.get_title(),
            requested_amount=self._applicant.get_requested_amount(),
            methodology=self._applicant.profile.methodology_full if "methodology" in revealed else None,
            budget_breakdown=self._applicant.profile.budget_full if "budget" in revealed else None,
            team_composition=self._applicant.profile.team_full if "team" in revealed else None,
            references=self._applicant.profile.references_full if "references" in revealed else None,
            clarification_response=None,
            actions_remaining=self._actions_remaining,
            sections_unlocked=revealed,
            step_number=self._step_count,
            last_action_error=self._last_error,
            evaluation_criteria=self._applicant.profile.evaluation_criteria,
        )
