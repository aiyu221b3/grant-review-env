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


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("GRANT_REVIEW_TASK", "easy") # setting default task to easy
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



def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    ) # we're turning it all to json format for simplicity


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    ) 


# idea here is to show the agent only the previous 4 actions to not overwhelm it. 
def build_user_prompt(observation_dict: dict, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_summary = json.dumps({
        k: v for k, v in observation_dict.items()
        if v is not None and k != "evaluation_criteria"
    }, indent=2) # cleaning up output for readability

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
        # llms wrap outputs in md, so we remove that and load json
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
        ) # this is for safety. if agent returns an invalid json, we reject.



def run_episode(task_name: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = GrantReviewEnv(task_name=task_name)
    # we start tracking state variables
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset() # initialize environment. we sample the true state from distribution.

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            # here the agent gets the observation as input, then it decides on an action.
            obs_dict = result.observation.model_dump() # emission function 
            action = get_agent_action(client, obs_dict, step, history)
            action_str = action.action_type.value
            # action is applied to environment, and we get the reward.
            result = env.step(action) 

            reward = result.reward or 0.0 # we get the reward.
            done = result.done
            error = result.info.get("error") 

            rewards.append(reward) # now, we track the metrics: rewards, steps taken, and error
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.2f}"
            ) # constructing ht here

            if done:
                break

        # now, we calculate the score! 
        max_possible = 1.0 + 0.2 + (0.25 * 2) + (0.15 * 4)  # correct decision + flaw bonus + confidence bonus
        raw_score = sum(rewards)
        score = raw_score / max_possible
        score = min(max(score, 0.001), 0.999) # bound the score to be between 0 and 1.
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards) # cleaning up env.


def main():
    """Run inference on all three tasks."""
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_episode(task_name=task) # running episode 


if __name__ == "__main__":
    main()
