# 📧 Email Triage OpenEnv

A real-world OpenEnv-compliant environment where an AI agent learns to
triage business emails — classifying, prioritizing, routing, and drafting replies.

---

## 🧠 Environment Description

Modern businesses receive thousands of emails daily. Triaging them correctly
is critical — a missed P1 support ticket or misrouted billing complaint has
real consequences. This environment simulates that exact challenge.

The agent processes a queue of business emails one at a time and must:
- Classify each email into the correct category
- Assign the right priority level
- Route it to the correct team
- Draft a professional reply (Task 3 only)

---

## 🗂️ Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | int | Current task: 1, 2, or 3 |
| `current_email` | object | Email with id, sender, subject, body, timestamp |
| `emails_remaining` | int | Emails left in the queue |
| `current_score` | float | Average score so far this episode |
| `step_number` | int | Current step index |
| `instruction` | string | Task-specific instruction for the agent |

---

## ⚡ Action Space

| Field | Type | Required for | Values |
|---|---|---|---|
| `category` | enum | All tasks | billing, support, spam, sales, hr, general |
| `priority` | enum | Task 2, 3 | P1 (Critical), P2 (High), P3 (Medium), P4 (Low) |
| `routing_team` | enum | Task 2, 3 | billing_team, support_team, sales_team, hr_team, spam_filter, general_team |
| `reply_draft` | string | Task 3 | Professional reply addressing sender's concern |

---

## 🏆 Tasks

| Task | Difficulty | Description | Success Threshold |
|---|---|---|---|
| Task 1 | Easy | Classify emails by category | 0.80 |
| Task 2 | Medium | Classify + prioritize + route | 0.75 |
| Task 3 | Hard | Full pipeline + draft reply | 0.65 |

---

## 🎯 Reward Function

Rewards are dense — given at every step, not just end of episode.

**Task 1 weights:**
- Category: 1.0 (binary)

**Task 2 weights:**
- Category: 0.40 (binary)
- Priority: 0.35 (partial credit — off by one level = 0.5)
- Routing: 0.25 (binary)

**Task 3 weights:**
- Category: 0.25 (binary)
- Priority: 0.25 (partial credit)
- Routing: 0.20 (binary)
- Reply: 0.30 (keyword coverage + length factor)
- Penalty: -0.3 for destructive replies

---

## 📦 Setup Instructions

### Local
```bash
# Clone the repo
git clone https://github.com/yourusername/email-triage-openenv
cd email-triage-openenv

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py
```

### Docker
```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key email-triage-openenv
```

---

## 🔌 API Usage
```python
import requests

BASE = "http://localhost:7860"

# Reset environment for task 1
obs = requests.post(f"{BASE}/reset", json={"task_id": 1}).json()

# Take a step
result = requests.post(f"{BASE}/step", json={
    "task_id": 1,
    "category": "billing",
}).json()

print(result["reward"])
print(result["done"])
```

---

## 📊 Baseline Scores

Evaluated using `gpt-4o-mini` with temperature=0, seed=42, 10 steps per task.

| Task | Model | Average Score |
|---|---|---|
| Task 1 - Classification | gpt-4o-mini | ~0.55 |
| Task 2 - Prioritization | gpt-4o-mini | ~0.45 |
| Task 3 - Full Triage | gpt-4o-mini | ~0.35 |

---

## 📁 Project Structure
```
email-triage-openenv/
├── env/
│   ├── email_env.py       # Core environment
│   ├── models.py          # Pydantic models
│   ├── email_data.py      # Email dataset
│   └── graders/           # Task graders
├── tasks/                 # Task YAML configs
├── baseline/              # Baseline inference script
├── app.py                 # FastAPI server
├── openenv.yaml           # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```