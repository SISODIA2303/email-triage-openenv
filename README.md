---
title: Email Triage OpenEnv
emoji: "ðŸ“§"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - email
  - triage
  - reinforcement-learning

---

# Email Triage OpenEnv

A real-world OpenEnv-compliant environment where an AI agent learns to
triage business emails - classifying, prioritizing, routing, and drafting replies.

## Environment Description

Modern businesses receive thousands of emails daily. Triaging them correctly
is critical. This environment simulates that exact challenge.

The agent processes a queue of business emails one at a time and must:
- Classify each email into the correct category
- Assign the right priority level
- Route it to the correct team
- Draft a professional reply (Task 3 only)

## Observation Space

| Field | Type | Description |
|---|---|---|
| task_id | int | Current task: 1, 2, or 3 |
| current_email | object | Email with id, sender, subject, body, timestamp |
| emails_remaining | int | Emails left in the queue |
| current_score | float | Average score so far this episode |
| step_number | int | Current step index |
| instruction | string | Task-specific instruction for the agent |

## Action Space

| Field | Type | Required for | Values |
|---|---|---|---|
| category | enum | All tasks | billing, support, spam, sales, hr, general |
| priority | enum | Task 2, 3 | P1 (Critical), P2 (High), P3 (Medium), P4 (Low) |
| routing_team | enum | Task 2, 3 | billing_team, support_team, sales_team, hr_team, spam_filter, general_team |
| reply_draft | string | Task 3 | Professional reply addressing sender's concern |

## Tasks

| Task | Difficulty | Description | Success Threshold |
|---|---|---|---|
| Task 1 | Easy | Classify emails by category | 0.80 |
| Task 2 | Medium | Classify + prioritize + route | 0.75 |
| Task 3 | Hard | Full pipeline + draft reply | 0.65 |

## Reward Function

Rewards are dense - given at every step, not just end of episode.

Task 1 weights:
- Category: 1.0 (binary)

Task 2 weights:
- Category: 0.40 (binary)
- Priority: 0.35 (partial credit - off by one level = 0.5)
- Routing: 0.25 (binary)

Task 3 weights:
- Category: 0.25 (binary)
- Priority: 0.25 (partial credit)
- Routing: 0.20 (binary)
- Reply: 0.30 (keyword coverage + length factor)
- Penalty: -0.3 for destructive replies

## Setup Instructions

### Local

    git clone https://github.com/siddharthsisodia23/email-triage-openenv
    cd email-triage-openenv
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 7860

### Docker

    docker build -t email-triage-openenv .
    docker run -p 7860:7860 email-triage-openenv

### Run Inference

    export HF_TOKEN=your_token_here
    python inference.py

## API Usage

    import requests
    BASE = "http://localhost:7860"
    obs = requests.post(f"{BASE}/reset", json={"task_id": 1}).json()
    result = requests.post(f"{BASE}/step", json={
        "task_id": 1,
        "category": "billing",
    }).json()

## Baseline Scores

Evaluated using Qwen2.5-72B-Instruct via HF router, temperature=0, seed=42.

| Task | Model | Average Score |
|---|---|---|
| Task 1 - Classification | Qwen2.5-72B | ~0.55 |
| Task 2 - Prioritization | Qwen2.5-72B | ~0.45 |
| Task 3 - Full Triage | Qwen2.5-72B | ~0.35 |

## Project Structure

    email-triage-openenv/
    |- env/
    |  |- email_env.py
    |  |- models.py
    |  |- email_data.py
    |  |- graders/
    |- tasks/
    |- baseline/
    |- server/
    |- app.py
    |- inference.py
    |- openenv.yaml
    |- Dockerfile
    |- requirements.txt
    |- README.md