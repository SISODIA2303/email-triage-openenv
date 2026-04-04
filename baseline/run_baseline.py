"""
Baseline inference script for Email Triage OpenEnv.
Uses OpenAI API to run an LLM agent against all 3 tasks.
Reads credentials from environment variables.
Produces reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline/run_baseline.py
"""

import os
import json
import time
from typing import Optional
from openai import OpenAI
from env.email_env import EmailTriageEnv
from env.models import Action, EmailCategory, Priority, RoutingTeam

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

MODEL = "gpt-4o-mini"           # cheap + capable enough for baseline
MAX_STEPS = 10                  # emails per episode
SEED = 42                       # reproducibility
DELAY_BETWEEN_CALLS = 0.5       # seconds, avoid rate limiting

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -----------------------------------------------------------------------
# SYSTEM PROMPTS
# -----------------------------------------------------------------------

SYSTEM_PROMPT_TASK1 = """
You are an expert email triage assistant.
You will receive an email and must classify it into exactly one category.

Categories: billing, support, spam, sales, hr, general

Respond ONLY with a valid JSON object in this exact format:
{
  "category": "<one of the categories above>"
}

No explanation. No extra text. Only the JSON object.
""".strip()

SYSTEM_PROMPT_TASK2 = """
You are an expert email triage assistant.
You will receive an email and must classify it, assign priority, and route it.

Categories: billing, support, spam, sales, hr, general
Priority: P1 (Critical), P2 (High), P3 (Medium), P4 (Low)
Routing teams: billing_team, support_team, sales_team, hr_team, spam_filter, general_team

Respond ONLY with a valid JSON object in this exact format:
{
  "category": "<category>",
  "priority": "<P1|P2|P3|P4>",
  "routing_team": "<routing_team>"
}

No explanation. No extra text. Only the JSON object.
""".strip()

SYSTEM_PROMPT_TASK3 = """
You are an expert email triage assistant.
You will receive an email and must classify it, assign priority, route it,
and draft a professional reply addressing the sender's concern.

Categories: billing, support, spam, sales, hr, general
Priority: P1 (Critical), P2 (High), P3 (Medium), P4 (Low)
Routing teams: billing_team, support_team, sales_team, hr_team, spam_filter, general_team

Reply guidelines:
- Be professional and empathetic
- Directly address the sender's concern
- Keep it concise but complete (at least 30 words)
- For spam emails, write an empty string for reply_draft

Respond ONLY with a valid JSON object in this exact format:
{
  "category": "<category>",
  "priority": "<P1|P2|P3|P4>",
  "routing_team": "<routing_team>",
  "reply_draft": "<your professional reply here>"
}

No explanation. No extra text. Only the JSON object.
""".strip()

SYSTEM_PROMPTS = {
    1: SYSTEM_PROMPT_TASK1,
    2: SYSTEM_PROMPT_TASK2,
    3: SYSTEM_PROMPT_TASK3,
}

# -----------------------------------------------------------------------
# AGENT HELPERS
# -----------------------------------------------------------------------

def build_user_message(observation) -> str:
    """Convert observation into a user message for the LLM."""
    email = observation.current_email
    return (
        f"From: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Timestamp: {email.timestamp}\n\n"
        f"Body:\n{email.body}\n\n"
        f"Instruction: {observation.instruction}"
    )


def call_llm(task_id: int, user_message: str) -> Optional[dict]:
    """Call OpenAI API and return parsed JSON response."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,    # deterministic for reproducibility
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"    [JSON parse error]: {e}")
        return None
    except Exception as e:
        print(f"    [API error]: {e}")
        return None


def parse_action(task_id: int, llm_output: Optional[dict]) -> Action:
    """
    Safely parse LLM output into an Action object.
    Falls back to defaults if parsing fails.
    """
    if llm_output is None:
        return Action(category=EmailCategory.GENERAL)

    try:
        category = EmailCategory(llm_output.get("category", "general").lower())
    except ValueError:
        category = EmailCategory.GENERAL

    priority = None
    if task_id in (2, 3) and "priority" in llm_output:
        try:
            priority = Priority(llm_output["priority"].upper())
        except ValueError:
            priority = Priority.P4

    routing_team = None
    if task_id in (2, 3) and "routing_team" in llm_output:
        try:
            routing_team = RoutingTeam(llm_output["routing_team"].lower())
        except ValueError:
            routing_team = RoutingTeam.GENERAL_TEAM

    reply_draft = None
    if task_id == 3:
        reply_draft = llm_output.get("reply_draft", "")

    return Action(
        category=category,
        priority=priority,
        routing_team=routing_team,
        reply_draft=reply_draft,
    )


# -----------------------------------------------------------------------
# EPISODE RUNNER
# -----------------------------------------------------------------------

def run_episode(task_id: int) -> dict:
    """Run one full episode for a given task. Returns summary stats."""
    print(f"\n{'='*60}")
    print(f"  TASK {task_id} — {['', 'Easy: Classification', 'Medium: Prioritization & Routing', 'Hard: Full Triage'][task_id]}")
    print(f"{'='*60}")

    env = EmailTriageEnv(task_id=task_id, max_steps=MAX_STEPS, seed=SEED)
    obs = env.reset()

    episode_rewards = []
    step = 0

    while True:
        step += 1
        email = obs.current_email
        print(f"\n  Step {step}/{MAX_STEPS} | Email: {email.email_id}")
        print(f"  Subject: {email.subject[:60]}...")

        # Build prompt and call LLM
        user_msg = build_user_message(obs)
        llm_output = call_llm(task_id, user_msg)
        action = parse_action(task_id, llm_output)

        print(f"  Action: category={action.category.value}", end="")
        if action.priority:
            print(f", priority={action.priority.value}", end="")
        if action.routing_team:
            print(f", routing={action.routing_team.value}", end="")
        print()

        # Step environment
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward.value)

        print(f"  Reward: {reward.value:.4f} | {reward.feedback}")

        time.sleep(DELAY_BETWEEN_CALLS)

        if done:
            break

    # Episode summary
    avg_score = sum(episode_rewards) / len(episode_rewards)
    print(f"\n  Episode complete.")
    print(f"  Steps: {len(episode_rewards)}")
    print(f"  Average Score: {avg_score:.4f}")
    print(f"  Min: {min(episode_rewards):.4f} | Max: {max(episode_rewards):.4f}")

    return {
        "task_id": task_id,
        "steps": len(episode_rewards),
        "average_score": round(avg_score, 4),
        "min_reward": round(min(episode_rewards), 4),
        "max_reward": round(max(episode_rewards), 4),
        "all_rewards": episode_rewards,
    }


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  EMAIL TRIAGE OPENENV — BASELINE INFERENCE")
    print(f"  Model: {MODEL} | Seed: {SEED} | Steps/task: {MAX_STEPS}")
    print("="*60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not set.")
        print("Run: export OPENAI_API_KEY=your_key_here")
        return

    all_results = []

    for task_id in [1, 2, 3]:
        result = run_episode(task_id)
        all_results.append(result)

    # Final report
    print("\n" + "="*60)
    print("  BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Task':<30} {'Avg Score':<12} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*60}")

    labels = {
        1: "Task 1 - Classification (Easy)",
        2: "Task 2 - Prioritization (Medium)",
        3: "Task 3 - Full Triage (Hard)",
    }

    for r in all_results:
        label = labels[r["task_id"]]
        print(f"  {label:<30} {r['average_score']:<12.4f} {r['min_reward']:<10.4f} {r['max_reward']:<10.4f}")

    overall = sum(r["average_score"] for r in all_results) / 3
    print(f"\n  Overall Baseline Score: {overall:.4f}")
    print("="*60)

    # Save results to JSON
    output_path = "baseline/baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL,
            "seed": SEED,
            "max_steps": MAX_STEPS,
            "results": all_results,
            "overall_score": round(overall, 4),
        }, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()