"""
Email Triage OpenEnv - Client

Usage (async):
    import asyncio
    from client import EmailTriageAction, EmailTriageClient
    from env.models import EmailCategory, Priority, RoutingTeam

    async def main():
        async with EmailTriageClient(base_url="https://siddharth-sisodia-email-triage-openenv.hf.space") as client:
            result = await client.reset(task_id=1)
            print(result)

            result = await client.step(EmailTriageAction(
                task_id=1,
                category=EmailCategory.BILLING,
            ))
            print(result)

    asyncio.run(main())

Usage (sync):
    from client import EmailTriageAction, EmailTriageClient
    from env.models import EmailCategory

    with EmailTriageClient(base_url="https://...hf.space").sync() as client:
        result = client.reset(task_id=1)
        result = client.step(EmailTriageAction(task_id=1, category=EmailCategory.BILLING))
        print(result)
"""

import asyncio
import httpx
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
# ASYNC CLIENT
# -----------------------------------------------------------------------

class EmailTriageClient:
    """
    Async client for the Email Triage OpenEnv environment.
    Use as an async context manager.
    """

    def __init__(self, base_url: str = "https://siddharth-sisodia-email-triage-openenv.hf.space"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_id: int = 1) -> dict:
        """Reset the environment and return the first observation."""
        self._check_client()
        response = await self._client.post(
            "/reset",
            json={"task_id": task_id},
        )
        response.raise_for_status()
        return response.json()

    async def step(self, action: EmailTriageAction) -> dict:
        """Take a step in the environment with the given action."""
        self._check_client()
        response = await self._client.post(
            "/step",
            json=action.dict(),
        )
        response.raise_for_status()
        return response.json()

    async def state(self, task_id: int = 1) -> dict:
        """Get the current state of the environment."""
        self._check_client()
        response = await self._client.get(
            "/state",
            params={"task_id": task_id},
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict:
        """Check if the environment server is healthy."""
        self._check_client()
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def _check_client(self):
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with EmailTriageClient() as client'"
            )

    def sync(self) -> "SyncEmailTriageClient":
        """Return a synchronous wrapper around this client."""
        return SyncEmailTriageClient(base_url=self.base_url)


# -----------------------------------------------------------------------
# SYNC WRAPPER
# -----------------------------------------------------------------------

class SyncEmailTriageClient:
    """
    Synchronous wrapper around EmailTriageClient.
    Use as a regular context manager.

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
        self._loop.run_until_complete(self._async_client.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loop.run_until_complete(
            self._async_client.__aexit__(exc_type, exc_val, exc_tb)
        )
        self._loop.close()

    def reset(self, task_id: int = 1) -> dict:
        return self._loop.run_until_complete(
            self._async_client.reset(task_id=task_id)
        )

    def step(self, action: EmailTriageAction) -> dict:
        return self._loop.run_until_complete(
            self._async_client.step(action)
        )

    def state(self, task_id: int = 1) -> dict:
        return self._loop.run_until_complete(
            self._async_client.state(task_id=task_id)
        )

    def health(self) -> dict:
        return self._loop.run_until_complete(
            self._async_client.health()
        )