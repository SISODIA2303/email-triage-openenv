"""
Inference Script - Email Triage OpenEnv
Meta OpenEnv Hackathon Submission
"""

import asyncio
import os
import json
import httpx
from typing import List, Optional
from openai import OpenAI

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = "https://siddharth-sisodia-email-triage-openenv.hf.space"

TASK_CONFIGS = [
    {"task_id": 1, "task_name": "email-classification",  "difficulty": "easy"},
    {"task_id": 2, "task_name": "email-prioritization",  "difficulty": "medium"},
    {"task_id": 3, "task_name": "email-full-triage",     "difficulty": "hard"},
]

BENCHMARK    = "email-triage-openenv"
MAX_STEPS    = 10
TEMPERATURE  = 0.0
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# -----------------------------------------------------------------------
# SYSTEM PROMPTS
# -----------------------------------------------------------------------

SYSTEM_PROMPTS = {
    1: """You are an expert email triage assistant.
Classify the email into exactly one category.
Categories: billing, support, spam, sales, hr, general
Respond ONLY with valid JSON: {"category": "<category>"}
No explanation. No markdown. Only JSON.""",

    2: """You are an expert email triage assistant.
Classify the email, assign priority, and route it.
Categories: billing, support, spam, sales, hr, general
Priority: P1 (Critical), P2 (High), P3 (Medium), P4 (Low)
Routing: billing_team, support_team, sales_team, hr_team, spam_filter, general_team
Respond ONLY with valid JSON:
{"category": "<category>", "priority": "<P1|P2|P3|P4>", "routing_team": "<team>"}
No explanation. No markdown. Only JSON.""",

    3: """You are an expert email triage assistant.
Classify, prioritize, route, and draft a professional reply.
Categories: billing, support, spam, sales, hr, general
Priority: P1 (Critical), P2 (High), P3 (Medium), P4 (Low)
Routing: billing_team, support_team, sales_team, hr_team, spam_filter, general_team
Respond ONLY with valid JSON:
{"category": "<category>", "priority": "<P1|P2|P3|P4>", "routing_team": "<team>", "reply_draft": "<reply>"}
No explanation. No markdown. Only JSON.""",
}

# -----------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# -----------------------------------------------------------------------
# ENV CLIENT (self-contained, no local imports)
# -----------------------------------------------------------------------

async def env_reset(client: httpx.AsyncClient, task_id: int = 1) -> dict:
    response = await client.post("/reset", json={"task_id": task_id}, timeout=30)
    response.raise_for_status()
    return response.json()

async def env_step(client: httpx.AsyncClient, action: dict) -> dict:
    response = await client.post("/step", json=action, timeout=30)
    response.raise_for_status()
    return response.json()

# -----------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------

def build_user_prompt(observation: dict) -> str:
    email = observation.get("current_email", {})
    return (
        f"From: {email.get('sender', 'unknown')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Body:\n{email.get('body', '')}\n\n"
        f"Instruction: {observation.get('instruction', '')}"
    )

def call_llm(llm_client, task_id: int, observation: dict) -> Optional[dict]:
    if llm_client is None:
        return None
    try:
        user_prompt = build_user_prompt(observation)
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return None

def parse_action(task_id: int, llm_output: Optional[dict]) -> dict:
    if not llm_output:
        return {"task_id": task_id, "category": "general"}

    action = {"task_id": task_id}

    # category
    cat = llm_output.get("category", "general").lower()
    if cat not in ["billing","support","spam","sales","hr","general"]:
        cat = "general"
    action["category"] = cat

    # priority
    if task_id in (2, 3):
        pri = llm_output.get("priority", "P4").upper()
        if pri not in ["P1","P2","P3","P4"]:
            pri = "P4"
        action["priority"] = pri

    # routing
    if task_id in (2, 3):
        teams = ["billing_team","support_team","sales_team","hr_team","spam_filter","general_team"]
        route = llm_output.get("routing_team", "general_team").lower()
        if route not in teams:
            route = "general_team"
        action["routing_team"] = route

    # reply
    if task_id == 3:
        action["reply_draft"] = llm_output.get("reply_draft", "")

    return action

def action_to_str(action: dict) -> str:
    parts = [f"category={action.get('category','general')}"]
    if "priority" in action:
        parts.append(f"priority={action['priority']}")
    if "routing_team" in action:
        parts.append(f"routing={action['routing_team']}")
    return "(" + ",".join(parts) + ")"

# -----------------------------------------------------------------------
# EPISODE RUNNER
# -----------------------------------------------------------------------

async def run_episode(
    http_client: httpx.AsyncClient,
    llm_client,
    task_config: dict,
) -> dict:
    task_id   = task_config["task_id"]
    task_name = task_config["task_name"]

    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        data       = await env_reset(http_client, task_id)
        observation = data["observation"]
        done        = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            error  = None
            action = parse_action(task_id, None)

            try:
                llm_out = call_llm(llm_client, task_id, observation)
                action  = parse_action(task_id, llm_out)
            except Exception as e:
                error = str(e)

            action_str = action_to_str(action)
            reward     = 0.0

            try:
                step_data   = await env_step(http_client, action)
                observation = step_data["observation"]
                reward      = step_data["reward"]["value"]
                done        = step_data["done"]
            except Exception as e:
                error = str(e)
                done  = True

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        log_step(step=1, action="(error)", reward=0.0, done=True, error=str(e))

    finally:
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards if rewards else [0.0],
        )

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "steps":     steps_taken,
        "score":     score,
        "success":   success,
        "rewards":   rewards,
    }

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

async def main() -> None:
    print("[DEBUG] Starting inference.py", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print("[DEBUG] LLM client initialized", flush=True)
    except Exception as e:
        print(f"[DEBUG] LLM client failed: {e}", flush=True)
        llm_client = None

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=60.0) as http_client:
        for task_config in TASK_CONFIGS:
            try:
                await run_episode(http_client, llm_client, task_config)
            except Exception as e:
                print(f"[DEBUG] Task {task_config['task_id']} outer error: {e}", flush=True)
                log_start(task=task_config["task_name"], env=BENCHMARK, model=MODEL_NAME)
                log_step(step=1, action="(error)", reward=0.0, done=True, error=str(e))
                log_end(success=False, steps=1, score=0.0, rewards=[0.0])


if __name__ == "__main__":
    asyncio.run(main())