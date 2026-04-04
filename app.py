"""
FastAPI wrapper around EmailTriageEnv.
Exposes reset / step / state as both HTTP and WebSocket endpoints.
Required for Hugging Face Spaces deployment and OpenEnv spec compliance.
"""

import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
# HTTP ENDPOINTS
# -----------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "tasks": [1, 2, 3],
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs", "/ws/{task_id}"],
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


# -----------------------------------------------------------------------
# WEBSOCKET ENDPOINT
# -----------------------------------------------------------------------

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: int):
    """
    WebSocket endpoint for OpenEnv spec compliance.

    Message format from client:
        {"command": "reset"}
        {"command": "step", "action": {"category": "billing", ...}}
        {"command": "state"}

    Response format:
        {"type": "observation", "data": {...}}
        {"type": "step_result", "data": {"observation": {}, "reward": {}, "done": false, "info": {}}}
        {"type": "state", "data": {...}}
        {"type": "error", "data": {"message": "..."}}
    """

    if task_id not in (1, 2, 3):
        await websocket.close(code=4000)
        return

    await websocket.accept()
    env = get_env(task_id)

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "data": {
            "task_id": task_id,
            "message": f"Connected to Email Triage OpenEnv â€” Task {task_id}",
            "commands": ["reset", "step", "state"],
        }
    })

    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON received"}
                })
                continue

            command = message.get("command")

            # --- RESET ---
            if command == "reset":
                obs = env.reset()
                await websocket.send_json({
                    "type": "observation",
                    "data": obs.dict(),
                })

            # --- STEP ---
            elif command == "step":
                action_data = message.get("action", {})
                try:
                    # Parse category
                    category = EmailCategory(
                        action_data.get("category", "general").lower()
                    )

                    # Parse priority
                    priority = None
                    if "priority" in action_data and action_data["priority"]:
                        priority = Priority(action_data["priority"].upper())

                    # Parse routing team
                    routing_team = None
                    if "routing_team" in action_data and action_data["routing_team"]:
                        routing_team = RoutingTeam(
                            action_data["routing_team"].lower()
                        )

                    # Parse reply
                    reply_draft = action_data.get("reply_draft", None)

                    action = Action(
                        category=category,
                        priority=priority,
                        routing_team=routing_team,
                        reply_draft=reply_draft,
                    )

                    obs, reward, done, info = env.step(action)

                    await websocket.send_json({
                        "type": "step_result",
                        "data": {
                            "observation": obs.dict(),
                            "reward": reward.dict(),
                            "done": done,
                            "info": info,
                        }
                    })

                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Invalid action: {str(e)}"}
                    })

                except RuntimeError as e:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": str(e)}
                    })

            # --- STATE ---
            elif command == "state":
                await websocket.send_json({
                    "type": "state",
                    "data": env.state(),
                })

            # --- UNKNOWN COMMAND ---
            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {
                        "message": f"Unknown command '{command}'. Use: reset, step, state"
                    }
                })

    except WebSocketDisconnect:
        print(f"Client disconnected from task {task_id}")