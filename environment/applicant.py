"""
applicant.py
Simulates a grant applicant who strategically withholds information.
The applicant is not lying — just not volunteering weaknesses.
Agents must ask exactly the right questions to surface hidden flaws.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class ApplicantProfile(BaseModel):
    """
    Full ground truth of the proposal.
    Agent never sees this directly — only what it unlocks.
    """
    title: str
    abstract: str
    requested_amount: float
    difficulty: str

    # Full sections — revealed only when requested
    methodology_full: str
    budget_full: Dict[str, float]
    team_full: List[str]
    references_full: List[str]

    # Hidden flaws — agent must detect these
    hidden_flaws: List[str]

    # Ground truth decision
    should_be_funded: bool
    correct_score: float  # 0.0 - 1.0

    # Evaluation rubric weights
    evaluation_criteria: Dict[str, float]

    # Clarification responses — keyed by keyword in question
    clarification_map: Dict[str, str]
    default_clarification: str


class StrategicApplicant:
    """
    Simulates an applicant who answers what's asked
    but never volunteers weaknesses unprompted.
    """

    def __init__(self, profile: ApplicantProfile):
        self.profile = profile
        self._sections_revealed: List[str] = []

    def get_abstract(self) -> str:
        return self.profile.abstract

    def get_title(self) -> str:
        return self.profile.title

    def get_requested_amount(self) -> float:
        return self.profile.requested_amount

    def reveal_methodology(self) -> str:
        """
        Returns methodology section.
        If there's a hidden flaw here, it's present but subtle.
        """
        self._sections_revealed.append("methodology")
        return self.profile.methodology_full

    def reveal_budget(self) -> Dict[str, float]:
        """
        Returns budget breakdown.
        Conflicts of interest hidden here require cross-referencing with team.
        """
        self._sections_revealed.append("budget")
        return self.profile.budget_full

    def reveal_team(self) -> List[str]:
        """
        Returns team composition.
        Cross-referencing with budget may reveal conflicts of interest.
        """
        self._sections_revealed.append("team")
        return self.profile.team_full

    def reveal_references(self) -> List[str]:
        self._sections_revealed.append("references")
        return self.profile.references_full

    def respond_to_clarification(self, question: str) -> str:
        """
        Applicant answers clarification questions strategically.
        Answers truthfully but minimally — doesn't volunteer extra information.
        If question hits a sensitive area, gives vague but technically true response.
        """
        question_lower = question.lower()
        for keyword, response in self.profile.clarification_map.items():
            if keyword.lower() in question_lower:
                return response
        return self.profile.default_clarification

    def get_hidden_flaws(self) -> List[str]:
        """Ground truth — only used by graders, never exposed to agent."""
        return self.profile.hidden_flaws

    def get_correct_decision(self) -> bool:
        """Ground truth — only used by graders."""
        return self.profile.should_be_funded

    def get_sections_revealed(self) -> List[str]:
        return self._sections_revealed.copy()

    def cross_reference_detectable(self) -> bool:
        """
        Returns True if agent has unlocked both budget AND team —
        meaning the conflict of interest is now detectable.
        Only relevant for hard task.
        """
        return (
            "budget" in self._sections_revealed
            and "team" in self._sections_revealed
        )
