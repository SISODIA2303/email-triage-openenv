"""
FastAPI wrapper around EmailTriageEnv.
Exposes reset / step / state as HTTP endpoints.
Required for Hugging Face Spaces deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from env.email_env import EmailTriageEnv
from env.models import Action, EmailCategory, Priority, RoutingTeam

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env instance per task stored in memory
_envs: dict = {}


def get_env(task_id: int) -> EmailTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id, max_steps=10, seed=42)
    return _envs[task_id]


# -----------------------------------------------------------------------
# REQUEST MODELS
# -----------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    task_id: int = 1
    category: EmailCategory
    priority: Optional[Priority] = None
    routing_team: Optional[RoutingTeam] = None
    reply_draft: Optional[str] = None


# -----------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "tasks": [1, 2, 3],
        "endpoints": ["/reset", "/step", "/state", "/docs"],
    }


@app.post("/reset")
def reset(request: ResetRequest):
    if request.task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
    env = get_env(request.task_id)
    obs = env.reset()
    return {"observation": obs.dict()}


@app.post("/step")
def step(request: StepRequest):
    if request.task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")

    env = get_env(request.task_id)

    action = Action(
        category=request.category,
        priority=request.priority,
        routing_team=request.routing_team,
        reply_draft=request.reply_draft,
    )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: int = 1):
    if task_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, or 3")
    env = get_env(task_id)
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}