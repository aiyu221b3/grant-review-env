"""
Inference Script — Grant Review Environment
============================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strict — do not deviate):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from environment.env import GrantReviewEnv
from environment.models import ActionType, GrantReviewAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("GRANT_REVIEW_TASK", "easy")
BENCHMARK = "grant-review-env"
MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert grant reviewer for a research funding committee.
    You must evaluate a research grant proposal and decide whether to fund it.

    You start with only the abstract. You can request more information before deciding.
    You have a limited action budget — use it wisely.

    Available actions (respond with JSON only):
    1. Request methodology:
       {"action_type": "request_methodology"}
    2. Request budget breakdown:
       {"action_type": "request_budget"}
    3. Request team composition:
       {"action_type": "request_team"}
    4. Request references:
       {"action_type": "request_references"}
    5. Ask a clarification question:
       {"action_type": "request_clarification", "clarification_question": "your question here"}
    6. Approve the proposal:
       {"action_type": "approve", "justification": "your reasoning", "confidence": 0.85}
    7. Reject the proposal:
       {"action_type": "reject", "justification": "your reasoning", "confidence": 0.85}

    Strategy:
    - Read the abstract carefully first.
    - Request sections most likely to reveal strengths or weaknesses.
    - Look for hidden flaws — missing methodology, budget conflicts, team issues.
    - Make your decision before running out of actions.
    - Respond ONLY with valid JSON. No explanation outside the JSON.
""").strip()


# ---------------------------------------------------------------------------
# Logging — strict format, do not modify
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def build_user_prompt(observation_dict: dict, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_summary = json.dumps({
        k: v for k, v in observation_dict.items()
        if v is not None and k != "evaluation_criteria"
    }, indent=2)

    return textwrap.dedent(f"""
        Step: {step}
        Actions remaining: {observation_dict.get('actions_remaining', '?')}

        Current proposal state:
        {obs_summary}

        Previous steps:
        {history_block}

        What is your next action? Respond with JSON only.
    """).strip()


def get_agent_action(
    client: OpenAI,
    observation_dict: dict,
    step: int,
    history: List[str]
) -> GrantReviewAction:
    """Call LLM to get next action. Falls back to REJECT on failure."""
    user_prompt = build_user_prompt(observation_dict, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action_data = json.loads(raw.strip())
        return GrantReviewAction(**action_data)
    except Exception as exc:
        print(f"[DEBUG] Agent action parse failed: {exc}", flush=True)
        return GrantReviewAction(
            action_type=ActionType.REJECT,
            justification="Failed to parse action — defaulting to reject.",
            confidence=0.1
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_episode(task_name: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = GrantReviewEnv(task_name=task_name)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = result.observation.model_dump()
            action = get_agent_action(client, obs_dict, step, history)
            action_str = action.action_type.value

            result = env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = result.info.get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.2f}"
            )

            if done:
                break

        # Score = cumulative reward normalized to [0, 1]
        # Max possible reward = CORRECT_DECISION + FLAW_BONUS + CONFIDENCE_BONUS
        max_possible = 1.0 + 0.2 + (0.25 * 2) + (0.15 * 4)  # roughly 2.5
        raw_score = sum(rewards)
        score = min(max(raw_score / max_possible, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    """Run inference on all three tasks."""
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_episode(task_name=task)


if __name__ == "__main__":
    main()
