"""
Inference Script — Email Triage OpenEnv
Meta OpenEnv Hackathon Submission

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
from typing import List, Optional
from openai import OpenAI
from client import EmailTriageClient, EmailTriageAction
from env.models import EmailCategory, Priority, RoutingTeam

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

# At the top of inference.py, update config section:
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_CONFIGS = [
    {"task_id": 1, "task_name": "email-classification",   "difficulty": "easy"},
    {"task_id": 2, "task_name": "email-prioritization",   "difficulty": "medium"},
    {"task_id": 3, "task_name": "email-full-triage",      "difficulty": "hard"},
]

BENCHMARK = "email-triage-openenv"
MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# -----------------------------------------------------------------------
# SYSTEM PROMPTS
# -----------------------------------------------------------------------

SYSTEM_PROMPTS = {
    1: """
You are an expert email triage assistant.
Classify the email into exactly one category.
Categories: billing, support, spam, sales, hr, general

Respond ONLY with valid JSON:
{"category": "<category>"}
No explanation. No markdown. Only JSON.
""".strip(),

    2: """
You are an expert email triage assistant.
Classify the email, assign priority, and route it.

Categories: billing, support, spam, sales, hr, general
Priority: P1 (Critical), P2 (High), P3 (Medium), P4 (Low)
Routing: billing_team, support_team, sales_team, hr_team, spam_filter, general_team

Respond ONLY with valid JSON:
{"category": "<category>", "priority": "<P1|P2|P3|P4>", "routing_team": "<team>"}
No explanation. No markdown. Only JSON.
""".strip(),

    3: """
You are an expert email triage assistant.
Classify the email, assign priority, route it, and draft a professional reply.

Categories: billing, support, spam, sales, hr, general
Priority: P1 (Critical), P2 (High), P3 (Medium), P4 (Low)
Routing: billing_team, support_team, sales_team, hr_team, spam_filter, general_team

Reply guidelines:
- Professional and empathetic tone
- Directly address the sender's concern
- Minimum 30 words
- For spam: use empty string for reply_draft

Respond ONLY with valid JSON:
{"category": "<category>", "priority": "<P1|P2|P3|P4>", "routing_team": "<team>", "reply_draft": "<reply>"}
No explanation. No markdown. Only JSON.
""".strip(),
}

# -----------------------------------------------------------------------
# LOGGING HELPERS
# -----------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# -----------------------------------------------------------------------
# LLM HELPERS
# -----------------------------------------------------------------------

def build_user_prompt(observation: dict) -> str:
    email = observation.get("current_email", {})
    return (
        f"From: {email.get('sender', 'unknown')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Timestamp: {email.get('timestamp', '')}\n\n"
        f"Body:\n{email.get('body', '')}\n\n"
        f"Instruction: {observation.get('instruction', '')}"
    )


def call_llm(client, task_id: int, observation: dict) -> Optional[dict]:
    if client is None:
        return None
    
    user_prompt = build_user_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return None


def parse_action(task_id: int, llm_output: Optional[dict]) -> EmailTriageAction:
    if llm_output is None:
        return EmailTriageAction(
            task_id=task_id,
            category=EmailCategory.GENERAL,
        )

    # Parse category
    try:
        category = EmailCategory(llm_output.get("category", "general").lower())
    except ValueError:
        category = EmailCategory.GENERAL

    # Parse priority
    priority = None
    if task_id in (2, 3) and "priority" in llm_output:
        try:
            priority = Priority(llm_output["priority"].upper())
        except ValueError:
            priority = Priority.P4

    # Parse routing team
    routing_team = None
    if task_id in (2, 3) and "routing_team" in llm_output:
        try:
            routing_team = RoutingTeam(llm_output["routing_team"].lower())
        except ValueError:
            routing_team = RoutingTeam.GENERAL_TEAM

    # Parse reply
    reply_draft = None
    if task_id == 3:
        reply_draft = llm_output.get("reply_draft", "")

    return EmailTriageAction(
        task_id=task_id,
        category=category,
        priority=priority,
        routing_team=routing_team,
        reply_draft=reply_draft,
    )


def action_to_str(action: EmailTriageAction) -> str:
    parts = [f"category={action.category.value}"]
    if action.priority:
        parts.append(f"priority={action.priority.value}")
    if action.routing_team:
        parts.append(f"routing={action.routing_team.value}")
    if action.reply_draft:
        # truncate reply for stdout
        reply_preview = action.reply_draft[:40].replace("\n", " ")
        parts.append(f"reply='{reply_preview}...'")
    return "(" + ",".join(parts) + ")"


# -----------------------------------------------------------------------
# EPISODE RUNNER
# -----------------------------------------------------------------------

async def run_episode(
    env: EmailTriageClient,
    llm_client,
    task_config: dict,
) -> dict:
    task_id = task_config["task_id"]
    task_name = task_config["task_name"]

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            observation = result.observation
            error = None

            try:
                llm_output = call_llm(llm_client, task_id, observation)
                action = parse_action(task_id, llm_output)
                action_str = action_to_str(action)
            except Exception as e:
                action = parse_action(task_id, None)
                action_str = action_to_str(action)
                error = str(e)

            try:
                result = await env.step(action)
                reward = result.reward
                done = result.done
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task_id": task_id,
        "task_name": task_name,
        "steps": steps_taken,
        "score": score,
        "success": success,
        "rewards": rewards,
    }

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

async def main() -> None:
    try:
        llm_client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception as e:
        print(f"[DEBUG] Failed to initialize LLM client: {e}", flush=True)
        # Still run episodes with fallback
        llm_client = None

    all_results = []

    for task_config in TASK_CONFIGS:
        try:
            if IMAGE_NAME:
                env = await EmailTriageClient.from_docker_image(IMAGE_NAME)
            else:
                env = EmailTriageClient(
                    base_url="https://siddharth-sisodia-email-triage-openenv.hf.space"
                )
                await env.__aenter__()

            try:
                result = await run_episode(env, llm_client, task_config)
                all_results.append(result)
            finally:
                try:
                    await env.close()
                except Exception as e:
                    print(f"[DEBUG] env.close() error: {e}", flush=True)

        except Exception as e:
            print(f"[DEBUG] Episode failed for task {task_config['task_id']}: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])

    if all_results:
        overall = sum(r["score"] for r in all_results) / len(all_results)
        print(f"\n[SUMMARY] overall_score={overall:.3f}", flush=True)
        for r in all_results:
            print(
                f"[SUMMARY] task={r['task_name']} "
                f"score={r['score']:.3f} "
                f"success={str(r['success']).lower()}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())