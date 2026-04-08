
from environment.env import GrantReviewEnv, GrantReviewAction, ActionType

# Test easy task
env = GrantReviewEnv(task_name="easy")
result = env.reset()
print("Abstract:", result.observation.abstract[:100])
print("Actions remaining:", result.observation.actions_remaining)

# Request methodology
action = GrantReviewAction(action_type=ActionType.REQUEST_METHODOLOGY)
result = env.step(action)
print("Reward:", result.reward)
print("Methodology unlocked:", result.observation.methodology[:50])

# Approve
action = GrantReviewAction(
    action_type=ActionType.APPROVE,
    justification="Strong proposal.",
    confidence=0.9
)
result = env.step(action)
print("Final reward:", result.reward)
print("Done:", result.done)

# Check grader
from environment.graders.easy import grade_easy
score = grade_easy(env.state())
print("Grader score:", score)

env = GrantReviewEnv(task_name="hard")
env.reset()
env.step(GrantReviewAction(action_type=ActionType.REQUEST_BUDGET))
env.step(GrantReviewAction(action_type=ActionType.REQUEST_TEAM))
print(env.state()["flaws_detected"])
# Should print: ['conflict_of_interest']