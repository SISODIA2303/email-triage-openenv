import random
from typing import Optional, Tuple, List
from env.models import (
    Observation, Action, Reward,
    EmailCategory, Priority, RoutingTeam
)
from env.email_data import generate_email_dataset, EMAIL_GROUND_TRUTH

class EmailTriageEnv:


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

        # Will be set on reset()
        self._email_queue: List = []
        self._current_index: int = 0
        self._step_number: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False

        # Load full dataset once
        self._all_emails = generate_email_dataset()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment for a new episode.
        Returns the first Observation.
        """
        random.seed(self.seed)

        # Shuffle and pick max_steps emails for this episode
        shuffled = self._all_emails.copy()
        random.shuffle(shuffled)
        self._email_queue = shuffled[:self.max_steps]

        self._current_index = 0
        self._step_number = 0
        self._total_reward = 0.0
        self._done = False

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """
        Process one agent action.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_email = self._email_queue[self._current_index]
        ground_truth = EMAIL_GROUND_TRUTH[current_email.email_id]

        # Grade the action
        reward = self._compute_reward(action, ground_truth)

        # Update state
        self._total_reward += reward.value
        self._step_number += 1
        self._current_index += 1

        # Check if episode is over
        if self._current_index >= len(self._email_queue):
            self._done = True

        info = {
            "email_id": current_email.email_id,
            "ground_truth": ground_truth,
            "action_taken": action.dict(),
            "reward_breakdown": reward.breakdown,
            "total_reward": self._total_reward,
            "step": self._step_number,
        }

        # If done, return final observation with empty email placeholder
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
        """
        Returns the current internal state of the environment.
        """
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
        """
        Compute reward based on task_id.
        Task 1: category only
        Task 2: category + priority + routing
        Task 3: category + priority + routing + reply
        """
        breakdown = {}
        feedback_parts = []

        # --- Category score (all tasks) ---
        category_correct = (action.category == ground_truth["category"])
        category_score = 1.0 if category_correct else 0.0
        breakdown["category"] = category_score
        expected_cat=ground_truth["category"].value
        feedback_parts.append(
            f"Category:{'✓ correct' if category_correct else f'✗ expected{expected_cat}'}"
        )

        if self.task_id == 1:
            total = category_score
            return Reward(
                value=round(total, 4),
                breakdown=breakdown,
                feedback=" | ".join(feedback_parts),
            )

        # --- Priority score (tasks 2 & 3) ---
        if action.priority is None:
            priority_score = 0.0
            feedback_parts.append("Priority: ✗ missing")
        else:
            priority_score = self._score_priority(action.priority, ground_truth["priority"])
            expected_pri=ground_truth["priority"].value
            feedback_parts.append(
                f"Priority:{'✓' if priority_score==1.0 else f'partial({priority_score}) expected{expected_pri}'}"
            )
        breakdown["priority"] = round(priority_score, 4)

        # --- Routing score (tasks 2 & 3) ---
        if action.routing_team is None:
            routing_score = 0.0
            feedback_parts.append("Routing: ✗ missing")
        else:
            routing_correct = (action.routing_team == ground_truth["routing_team"])
            routing_score = 1.0 if routing_correct else 0.0
            expected_route=ground_truth["routing_team"].value
            feedback_parts.append(
                f"Routing:{'✓' if routing_correct else f'✗ expected{expected_route}'}"
            )
        breakdown["routing"] = round(routing_score, 4)

        if self.task_id == 2:
            total = (category_score * 0.4) + (priority_score * 0.35) + (routing_score * 0.25)
            return Reward(
                value=round(total, 4),
                breakdown=breakdown,
                feedback=" | ".join(feedback_parts),
            )

        # --- Reply score (task 3 only) ---
        if action.reply_draft is None or action.reply_draft.strip() == "":
            reply_score = 0.0
            feedback_parts.append("Reply: ✗ missing")
        else:
            reply_score = self._score_reply(action.reply_draft, ground_truth)
            feedback_parts.append(f"Reply: {round(reply_score, 2)} keyword coverage")
        breakdown["reply"] = round(reply_score, 4)

        # Task 3 weights
        total = (
            (category_score * 0.25) +
            (priority_score * 0.25) +
            (routing_score * 0.20) +
            (reply_score * 0.30)
        )

        # Penalize clearly bad behavior
        if self._is_destructive_action(action):
            total = max(0.0, total - 0.3)
            feedback_parts.append("⚠ penalty: destructive reply detected")

        return Reward(
            value=round(total, 4),
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts),
        )

    def _score_priority(self, predicted: Priority, actual: Priority) -> float:
        """
        Partial credit for priority — being off by one level = 0.5
        """
        order = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]
        pred_idx = order.index(predicted)
        actual_idx = order.index(actual)
        diff = abs(pred_idx - actual_idx)

        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5
        else:
            return 0.0

    def _score_reply(self, reply: str, ground_truth: dict) -> float:
        """
        Score reply based on keyword coverage.
        Bonus for length (shows effort), penalty for very short replies.
        """
        keywords = ground_truth.get("reply_keywords", [])

        if not keywords:
            # Spam emails — no reply expected, penalize if reply is given
            return 0.0 if len(reply.strip()) > 10 else 1.0

        reply_lower = reply.lower()
        matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
        keyword_score = matched / len(keywords)

        # Length bonus: replies under 20 words get penalized
        word_count = len(reply.split())
        length_factor = min(1.0, word_count / 30)

        return round((keyword_score * 0.7) + (length_factor * 0.3), 4)

    def _is_destructive_action(self, action: Action) -> bool:
        """
        Detects clearly undesirable agent behavior.
        e.g. writing an aggressive or dismissive reply.
        """
        if action.reply_draft is None:
            return False
        bad_phrases = [
            "ignore", "delete this", "i don't care",
            "not my problem", "go away", "this is spam",
        ]
        reply_lower = action.reply_draft.lower()
        return any(phrase in reply_lower for phrase in bad_phrases)

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