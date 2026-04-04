"""
Email Triage OpenEnv - Client
Async and Sync client with hackathon-compliant from_docker_image() support.
"""

import asyncio
import httpx
import subprocess
import time
from typing import Optional
from env.models import EmailCategory, Priority, RoutingTeam
from pydantic import BaseModel


# -----------------------------------------------------------------------
# ACTION MODEL
# -----------------------------------------------------------------------

class EmailTriageAction(BaseModel):
    task_id: int = 1
    category: EmailCategory
    priority: Optional[Priority] = None
    routing_team: Optional[RoutingTeam] = None
    reply_draft: Optional[str] = None


# -----------------------------------------------------------------------
# STEP RESULT MODEL
# -----------------------------------------------------------------------

class StepResult(BaseModel):
    model_config={"arbitrary_types_allowed":True}

    observation:dict
    reward:float
    done:bool
    info:dict={}


# -----------------------------------------------------------------------
# ASYNC CLIENT
# -----------------------------------------------------------------------

class EmailTriageClient:
    """
    Async client for the Email Triage OpenEnv environment.

    Usage (remote HF Space):
        async with EmailTriageClient(base_url="https://...hf.space") as client:
            result = await client.reset(task_id=1)
            result = await client.step(action)
            await client.close()

    Usage (local docker):
        client = await EmailTriageClient.from_docker_image("email-triage-openenv")
        result = await client.reset(task_id=1)
        await client.close()
    """

    def __init__(
        self,
        base_url: str = "https://siddharth-sisodia-email-triage-openenv.hf.space",
    ):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._container_id: Optional[str] = None
        self._task_id: int = 1

    # ------------------------------------------------------------------
    # CONTEXT MANAGER
    # ------------------------------------------------------------------

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ------------------------------------------------------------------
    # FACTORY — from docker image
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: int = 7860,
        startup_wait: int = 10,
    ) -> "EmailTriageClient":
        """
        Spin up a local Docker container and connect to it.

        Args:
            image_name: Docker image name (e.g. 'email-triage-openenv')
            port: Host port to bind (default 7860)
            startup_wait: Seconds to wait for container startup
        """
        print(f"[DEBUG] Starting docker container from image: {image_name}", flush=True)

        result = subprocess.run(
            [
                "docker", "run", "-d",
                "-p", f"{port}:7860",
                image_name,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start docker container: {result.stderr}"
            )

        container_id = result.stdout.strip()
        print(f"[DEBUG] Container started: {container_id[:12]}", flush=True)

        # Wait for server to be ready
        base_url = f"http://localhost:{port}"
        client = cls(base_url=base_url)
        client._container_id = container_id
        client._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=60.0,
        )

        # Poll health endpoint until ready
        for attempt in range(startup_wait):
            try:
                response = await client._client.get("/health")
                if response.status_code == 200:
                    print(f"[DEBUG] Server ready after {attempt+1}s", flush=True)
                    return client
            except Exception:
                pass
            await asyncio.sleep(1)

        raise RuntimeError(
            f"Server did not become ready after {startup_wait}s"
        )

    # ------------------------------------------------------------------
    # CORE API
    # ------------------------------------------------------------------

    async def reset(self, task_id: int = 1) -> StepResult:
        """Reset the environment and return first observation."""
        self._check_client()
        self._task_id = task_id

        response = await self._client.post(
            "/reset",
            json={"task_id": task_id},
        )
        response.raise_for_status()
        data = response.json()

        return StepResult(
            observation=data["observation"],
            reward=0.0,
            done=False,
            info={},
        )

    async def step(self, action: EmailTriageAction) -> StepResult:
        """Take a step in the environment."""
        self._check_client()

        response = await self._client.post(
            "/step",
            json=action.dict(),
        )
        response.raise_for_status()
        data = response.json()

        return StepResult(
            observation=data["observation"],
            reward=data["reward"]["value"],
            done=data["done"],
            info=data.get("info", {}),
        )

    async def state(self, task_id: int = 1) -> dict:
        """Get current environment state."""
        self._check_client()
        response = await self._client.get(
            "/state",
            params={"task_id": task_id},
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict:
        """Check server health."""
        self._check_client()
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close HTTP client and stop docker container if running."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._container_id:
            print(
                f"[DEBUG] Stopping container {self._container_id[:12]}",
                flush=True,
            )
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
            )
            self._container_id = None

    def _check_client(self):
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with EmailTriageClient()' "
                "or 'await EmailTriageClient.from_docker_image()'"
            )

    def sync(self) -> "SyncEmailTriageClient":
        """Return synchronous wrapper."""
        return SyncEmailTriageClient(base_url=self.base_url)


# -----------------------------------------------------------------------
# SYNC WRAPPER
# -----------------------------------------------------------------------

class SyncEmailTriageClient:
    """
    Synchronous wrapper around EmailTriageClient.

    with EmailTriageClient(base_url="...").sync() as client:
        result = client.reset(task_id=1)
        result = client.step(action)
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._async_client = EmailTriageClient(base_url=base_url)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(
            self._async_client.__aenter__()
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loop.run_until_complete(
            self._async_client.__aexit__(exc_type, exc_val, exc_tb)
        )
        self._loop.close()

    def reset(self, task_id: int = 1) -> StepResult:
        return self._loop.run_until_complete(
            self._async_client.reset(task_id=task_id)
        )

    def step(self, action: EmailTriageAction) -> StepResult:
        return self._loop.run_until_complete(
            self._async_client.step(action)
        )

    def state(self, task_id: int = 1) -> dict:
        return self._loop.run_until_complete(
            self._async_client.state(task_id=task_id)
        )

    def close(self):
        self._loop.run_until_complete(
            self._async_client.close()
        )