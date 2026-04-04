"""
Email Triage OpenEnv
====================
A real-world OpenEnv-compliant email triage environment for AI agents.

Quick Start (async):
    import asyncio
    from email_triage_openenv import EmailTriageClient, EmailTriageAction
    from env.models import EmailCategory, Priority, RoutingTeam

    async def main():
        async with EmailTriageClient(
            base_url="https://siddharth-sisodia-email-triage-openenv.hf.space"
        ) as client:
            # Reset environment for task 1
            result = await client.reset(task_id=1)
            print(result)

            # Take a step
            result = await client.step(EmailTriageAction(
                task_id=1,
                category=EmailCategory.BILLING,
            ))
            print(result)

    asyncio.run(main())

Quick Start (sync):
    from email_triage_openenv import EmailTriageClient, EmailTriageAction
    from env.models import EmailCategory

    with EmailTriageClient(
        base_url="https://siddharth-sisodia-email-triage-openenv.hf.space"
    ).sync() as client:
        result = client.reset(task_id=1)
        result = client.step(EmailTriageAction(
            task_id=1,
            category=EmailCategory.BILLING,
        ))
        print(result)
"""

# Core environment
from env.email_env import EmailTriageEnv

# Pydantic models
from env.models import (
    Email,
    Observation,
    Action,
    Reward,
    EmailCategory,
    Priority,
    RoutingTeam,
)

# Graders
from env.graders.task1_grader import Task1Grader
from env.graders.task2_grader import Task2Grader
from env.graders.task3_grader import Task3Grader

# Client
from client import EmailTriageClient, EmailTriageAction

__version__ = "1.0.0"
__all__ = [
    # Environment
    "EmailTriageEnv",
    # Models
    "Email",
    "Observation",
    "Action",
    "Reward",
    "EmailCategory",
    "Priority",
    "RoutingTeam",
    # Graders
    "Task1Grader",
    "Task2Grader",
    "Task3Grader",
    # Client
    "EmailTriageClient",
    "EmailTriageAction",
]