"""
Basic tests for Email Triage OpenEnv.
Run with: pytest tests/test_env.py -v
"""
import pytest
from env.email_env import EmailTriageEnv
from env.models import Action, EmailCategory, Priority, RoutingTeam
from env.graders.task1_grader import Task1Grader
from env.graders.task2_grader import Task2Grader
from env.graders.task3_grader import Task3Grader
from env.email_data import generate_email_dataset, EMAIL_GROUND_TRUTH


# -----------------------------------------------------------------------
# DATASET TESTS
# -----------------------------------------------------------------------

def test_dataset_generates_emails():
    emails = generate_email_dataset()
    assert len(emails) > 0

def test_ground_truth_populated():
    generate_email_dataset()
    assert len(EMAIL_GROUND_TRUTH) > 0

def test_all_emails_have_ground_truth():
    emails = generate_email_dataset()
    for email in emails:
        assert email.email_id in EMAIL_GROUND_TRUTH


# -----------------------------------------------------------------------
# ENVIRONMENT TESTS
# -----------------------------------------------------------------------

def test_reset_returns_observation():
    env = EmailTriageEnv(task_id=1, max_steps=5)
    obs = env.reset()
    assert obs.task_id == 1
    assert obs.current_email is not None
    assert obs.emails_remaining >= 0
    assert obs.step_number == 0

def test_reset_cleans_state():
    env = EmailTriageEnv(task_id=1, max_steps=5)
    env.reset()
    action = Action(category=EmailCategory.BILLING)
    env.step(action)
    env.reset()
    state = env.state()
    assert state["step_number"] == 0
    assert state["total_reward"] == 0.0

def test_state_returns_dict():
    env = EmailTriageEnv(task_id=1, max_steps=5)
    env.reset()
    state = env.state()
    assert isinstance(state, dict)
    assert "task_id" in state
    assert "step_number" in state
    assert "done" in state

def test_step_returns_tuple():
    env = EmailTriageEnv(task_id=1, max_steps=5)
    env.reset()
    action = Action(category=EmailCategory.BILLING)
    obs, reward, done, info = env.step(action)
    assert obs is not None
    assert isinstance(reward.value, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_reward_in_valid_range():
    env = EmailTriageEnv(task_id=1, max_steps=5)
    env.reset()
    action = Action(category=EmailCategory.BILLING)
    _, reward, _, _ = env.step(action)
    assert -1.0 <= reward.value <= 1.0

def test_episode_ends_after_max_steps():
    env = EmailTriageEnv(task_id=1, max_steps=3)
    env.reset()
    action = Action(category=EmailCategory.BILLING)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(action)
        steps += 1
    assert steps == 3

def test_done_raises_on_extra_step():
    env = EmailTriageEnv(task_id=1, max_steps=2)
    env.reset()
    action = Action(category=EmailCategory.BILLING)
    env.step(action)
    env.step(action)
    with pytest.raises(RuntimeError):
        env.step(action)


# -----------------------------------------------------------------------
# TASK SPECIFIC TESTS
# -----------------------------------------------------------------------

def test_task1_only_needs_category():
    env = EmailTriageEnv(task_id=1, max_steps=3)
    env.reset()
    action = Action(category=EmailCategory.SUPPORT)
    _, reward, _, _ = env.step(action)
    assert reward.value >= 0.0

def test_task2_rewards_complete_action():
    env = EmailTriageEnv(task_id=2, max_steps=3)
    env.reset()
    action = Action(
        category=EmailCategory.SUPPORT,
        priority=Priority.P1,
        routing_team=RoutingTeam.SUPPORT_TEAM,
    )
    _, reward, _, _ = env.step(action)
    assert reward.value >= 0.0
    assert "priority" in reward.breakdown
    assert "routing" in reward.breakdown

def test_task2_penalizes_missing_fields():
    env = EmailTriageEnv(task_id=2, max_steps=3)
    env.reset()
    action = Action(category=EmailCategory.SUPPORT)
    _, reward, _, _ = env.step(action)
    assert reward.breakdown.get("priority", 0) == 0.0
    assert reward.breakdown.get("routing", 0) == 0.0

def test_task3_rewards_reply():
    env = EmailTriageEnv(task_id=3, max_steps=3)
    env.reset()
    action = Action(
        category=EmailCategory.BILLING,
        priority=Priority.P1,
        routing_team=RoutingTeam.BILLING_TEAM,
        reply_draft="We apologize for the inconvenience and will resolve your billing issue promptly.",
    )
    _, reward, _, _ = env.step(action)
    assert "reply" in reward.breakdown


# -----------------------------------------------------------------------
# GRADER TESTS
# -----------------------------------------------------------------------

def test_task1_grader_correct():
    generate_email_dataset()
    grader = Task1Grader()
    email_id = list(EMAIL_GROUND_TRUTH.keys())[0]
    correct_category = EMAIL_GROUND_TRUTH[email_id]["category"]
    action = Action(category=correct_category)
    reward = grader.grade(email_id, action)
    assert reward.value == 1.0

def test_task1_grader_wrong():
    generate_email_dataset()
    grader = Task1Grader()
    email_id = list(EMAIL_GROUND_TRUTH.keys())[0]
    action = Action(category=EmailCategory.SPAM)
    reward = grader.grade(email_id, action)
    assert 0.0 <= reward.value <= 1.0

def test_task2_grader_score_range():
    generate_email_dataset()
    grader = Task2Grader()
    email_id = list(EMAIL_GROUND_TRUTH.keys())[0]
    action = Action(
        category=EmailCategory.BILLING,
        priority=Priority.P2,
        routing_team=RoutingTeam.BILLING_TEAM,
    )
    reward = grader.grade(email_id, action)
    assert 0.0 <= reward.value <= 1.0

def test_task3_grader_score_range():
    generate_email_dataset()
    grader = Task3Grader()
    email_id = list(EMAIL_GROUND_TRUTH.keys())[0]
    action = Action(
        category=EmailCategory.BILLING,
        priority=Priority.P1,
        routing_team=RoutingTeam.BILLING_TEAM,
        reply_draft="We are sorry for the trouble and will resolve your billing issue as soon as possible.",
    )
    reward = grader.grade(email_id, action)
    assert 0.0 <= reward.value <= 1.0

def test_grader_unknown_email_id():
    grader = Task1Grader()
    action = Action(category=EmailCategory.BILLING)
    reward = grader.grade("UNKNOWN_ID", action)
    assert reward.value == 0.0