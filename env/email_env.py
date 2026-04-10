import random
from typing import Optional, Tuple, List
from env.models import (
    Observation, Action, Reward,
    EmailCategory, Priority, RoutingTeam
)
from env.email_data import generate_email_dataset, EMAIL_GROUND_TRUTH


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage Environment.

    Task 1 (Easy)   - Classify emails by category only
    Task 2 (Medium) - Classify + assign priority + route to team
    Task 3 (Hard)   - Classify + priority + route + draft reply

    All reward values are strictly between 0.001 and 0.999.
    """

    TASK_INSTRUCTIONS = {
        1: (
            "TASK 1 - CLASSIFICATION: Read the email and classify it into one of: "
            "billing, support, spam, sales, hr, general. "
            "Only the 'category' field is required."
        ),
        2: (
            "TASK 2 - PRIORITIZATION & ROUTING: Read the email and provide: "
            "1) category  2) priority (P1/P2/P3/P4)  3) routing_team. "
            "All three fields are required."
        ),
        3: (
            "TASK 3 - FULL TRIAGE: Read the email and provide: "
            "1) category  2) priority  3) routing_team  4) reply_draft. "
            "Write a professional reply addressing the sender's concern. "
            "All four fields are required."
        ),
    }

    def __init__(self, task_id: int = 1, max_steps: int = 10, seed: int = 42):
        assert task_id in (1, 2, 3), "task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.max_steps = max_steps
        self.seed = seed

        self._email_queue: List = []
        self._current_index: int = 0
        self._step_number: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False

        self._all_emails = generate_email_dataset()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        random.seed(self.seed)
        shuffled = self._all_emails.copy()
        random.shuffle(shuffled)
        self._email_queue = shuffled[:self.max_steps]
        self._current_index = 0
        self._step_number = 0
        self._total_reward = 0.0
        self._done = False
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_email = self._email_queue[self._current_index]
        ground_truth = EMAIL_GROUND_TRUTH[current_email.email_id]

        reward = self._compute_reward(action, ground_truth)

        # Final safety clamp — strictly between 0.001 and 0.999
        reward.value = max(0.001, min(0.999, round(reward.value, 4)))

        self._total_reward += reward.value
        self._step_number += 1
        self._current_index += 1

        if self._current_index >= len(self._email_queue):
            self._done = True

        info = {
            "email_id": current_email.email_id,
            "ground_truth": {
                "category": ground_truth["category"].value,
                "priority": ground_truth["priority"].value,
                "routing_team": ground_truth["routing_team"].value,
            },
            "action_taken": action.model_dump(),
            "reward_breakdown": reward.breakdown,
            "total_reward": self._total_reward,
            "step": self._step_number,
        }

        if self._done:
            final_obs = Observation(
                task_id=self.task_id,
                current_email=current_email,
                emails_remaining=0,
                current_score=round(self._total_reward / self._step_number, 4),
                step_number=self._step_number,
                instruction="Episode complete.",
            )
            return final_obs, reward, True, info

        return self._make_observation(), reward, False, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_number": self._step_number,
            "current_index": self._current_index,
            "total_reward": round(self._total_reward, 4),
            "average_score": round(
                self._total_reward / self._step_number, 4
            ) if self._step_number > 0 else 0.0,
            "emails_remaining": max(0, len(self._email_queue) - self._current_index),
            "done": self._done,
            "queue_size": len(self._email_queue),
        }

    # ------------------------------------------------------------------
    # REWARD COMPUTATION
    # ------------------------------------------------------------------

    def _compute_reward(self, action: Action, ground_truth: dict) -> Reward:
        breakdown = {}
        feedback_parts = []

        # --- Category score (all tasks) ---
        category_correct = (action.category == ground_truth["category"])
        category_score = 0.999 if category_correct else 0.001
        breakdown["category"] = category_score
        expected_cat = ground_truth["category"].value
        feedback_parts.append(
            f"Category: correct ({action.category.value})"
            if category_correct
            else f"Category: got '{action.category.value}' expected '{expected_cat}'"
        )

        if self.task_id == 1:
            total = max(0.001, min(0.999, category_score))
            return Reward(
                value=round(total, 4),
                breakdown=breakdown,
                feedback=" | ".join(feedback_parts),
            )

        # --- Priority score (tasks 2 & 3) ---
        if action.priority is None:
            priority_score = 0.001
            feedback_parts.append("Priority: missing")
        else:
            priority_score = self._score_priority(
                action.priority, ground_truth["priority"]
            )
            expected_pri = ground_truth["priority"].value
            feedback_parts.append(
                f"Priority: correct ({action.priority.value})"
                if priority_score >= 0.999
                else f"Priority: partial={priority_score} got '{action.priority.value}' expected '{expected_pri}'"
            )
        breakdown["priority"] = round(priority_score, 4)

        # --- Routing score (tasks 2 & 3) ---
        if action.routing_team is None:
            routing_score = 0.001
            feedback_parts.append("Routing: missing")
        else:
            routing_correct = (action.routing_team == ground_truth["routing_team"])
            routing_score = 0.999 if routing_correct else 0.001
            expected_route = ground_truth["routing_team"].value
            feedback_parts.append(
                f"Routing: correct ({action.routing_team.value})"
                if routing_correct
                else f"Routing: got '{action.routing_team.value}' expected '{expected_route}'"
            )
        breakdown["routing"] = round(routing_score, 4)

        if self.task_id == 2:
            total = (
                category_score * 0.40 +
                priority_score * 0.35 +
                routing_score  * 0.25
            )
            total = max(0.001, min(0.999, round(total, 4)))
            return Reward(
                value=total,
                breakdown=breakdown,
                feedback=" | ".join(feedback_parts),
            )

        # --- Reply score (task 3 only) ---
        if action.reply_draft is None or action.reply_draft.strip() == "":
            reply_score = 0.001
            feedback_parts.append("Reply: missing")
        else:
            reply_score = self._score_reply(action.reply_draft, ground_truth)
            reply_score = max(0.001, min(0.999, reply_score))
            feedback_parts.append(f"Reply: {round(reply_score, 2)} score")
        breakdown["reply"] = round(reply_score, 4)

        # Penalty
        penalty = 0.0
        if self._is_destructive_action(action):
            penalty = 0.3
            feedback_parts.append("penalty: destructive reply")

        total = (
            category_score * 0.25 +
            priority_score * 0.25 +
            routing_score  * 0.20 +
            reply_score    * 0.30
        )
        total = max(0.0, total - penalty)
        total = max(0.001, min(0.999, round(total, 4)))

        return Reward(
            value=total,
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts),
        )

    def _score_priority(self, predicted: Priority, actual: Priority) -> float:
        order = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]
        diff = abs(order.index(predicted) - order.index(actual))
        if diff == 0:
            return 0.999
        elif diff == 1:
            return 0.5
        return 0.001

    def _score_reply(self, reply: str, ground_truth: dict) -> float:
        keywords = ground_truth.get("reply_keywords", [])
        if not keywords:
            return 0.001 if len(reply.strip()) > 10 else 0.999
        reply_lower = reply.lower()
        matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
        keyword_score = matched / len(keywords)
        word_count = len(reply.split())
        length_factor = min(0.999, word_count / 30)
        score = (keyword_score * 0.7) + (length_factor * 0.3)
        return round(max(0.001, min(0.999, score)), 4)

    def _is_destructive_action(self, action: Action) -> bool:
        if action.reply_draft is None:
            return False
        bad_phrases = [
            "ignore", "delete this", "i don't care",
            "not my problem", "go away", "this is spam",
        ]
        return any(phrase in action.reply_draft.lower() for phrase in bad_phrases)

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        current_email = self._email_queue[self._current_index]
        emails_remaining = len(self._email_queue) - self._current_index - 1
        return Observation(
            task_id=self.task_id,
            current_email=current_email,
            emails_remaining=emails_remaining,
            current_score=round(
                self._total_reward / self._step_number, 4
            ) if self._step_number > 0 else 0.0,
            step_number=self._step_number,
            instruction=self.TASK_INSTRUCTIONS[self.task_id],
        )