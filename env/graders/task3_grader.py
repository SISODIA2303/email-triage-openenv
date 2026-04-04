from env.models import Action, Reward, Priority
from env.email_data import EMAIL_GROUND_TRUTH


class Task3Grader:
    """
    Task 3 - HARD: Full triage pipeline.
    Weights: category=0.25, priority=0.25, routing=0.20, reply=0.30
    Reply scored on keyword coverage + length factor.
    """

    WEIGHTS = {
        "category": 0.25,
        "priority": 0.25,
        "routing": 0.20,
        "reply": 0.30,
    }

    def grade(self, email_id: str, action: Action) -> Reward:
        if email_id not in EMAIL_GROUND_TRUTH:
            return Reward(
                value=0.0,
                breakdown={"category": 0.0, "priority": 0.0, "routing": 0.0, "reply": 0.0},
                feedback=f"Unknown email_id: {email_id}",
            )

        ground_truth = EMAIL_GROUND_TRUTH[email_id]
        breakdown = {}
        feedback_parts = []

        # --- Category ---
        category_correct = action.category == ground_truth["category"]
        category_score = 1.0 if category_correct else 0.0
        breakdown["category"] = category_score
        expected_cat = ground_truth["category"].value
        feedback_parts.append(
            f"Category: ✓ ({action.category.value})"
            if category_correct
            else f"Category: ✗ got '{action.category.value}' expected '{expected_cat}'"
        )

        # --- Priority ---
        if action.priority is None:
            priority_score = 0.0
            feedback_parts.append("Priority: ✗ missing")
        else:
            priority_score = self._score_priority(
                action.priority, ground_truth["priority"]
            )
            expected_pri = ground_truth["priority"].value
            feedback_parts.append(
                f"Priority: ✓ ({action.priority.value})"
                if priority_score == 1.0
                else f"Priority: partial={priority_score} got '{action.priority.value}' expected '{expected_pri}'"
            )
        breakdown["priority"] = round(priority_score, 4)

        # --- Routing ---
        if action.routing_team is None:
            routing_score = 0.0
            feedback_parts.append("Routing: ✗ missing")
        else:
            routing_correct = action.routing_team == ground_truth["routing_team"]
            routing_score = 1.0 if routing_correct else 0.0
            expected_route = ground_truth["routing_team"].value
            feedback_parts.append(
                f"Routing: ✓ ({action.routing_team.value})"
                if routing_correct
                else f"Routing: ✗ got '{action.routing_team.value}' expected '{expected_route}'"
            )
        breakdown["routing"] = round(routing_score, 4)

        # --- Reply ---
        if action.reply_draft is None or action.reply_draft.strip() == "":
            reply_score = 0.0
            feedback_parts.append("Reply: ✗ missing")
        else:
            reply_score = self._score_reply(
                action.reply_draft, ground_truth
            )
            feedback_parts.append(f"Reply: {round(reply_score, 2)} score")
        breakdown["reply"] = round(reply_score, 4)

        # --- Penalty for destructive reply ---
        penalty = 0.0
        if self._is_destructive(action.reply_draft):
            penalty = 0.3
            feedback_parts.append("⚠ penalty: destructive reply detected")

        total = (
            category_score * self.WEIGHTS["category"] +
            priority_score * self.WEIGHTS["priority"] +
            routing_score * self.WEIGHTS["routing"] +
            reply_score * self.WEIGHTS["reply"]
        )
        total = max(0.0, round(total - penalty, 4))

        return Reward(
            value=total,
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts),
        )

    def grade_episode(self, results: list) -> dict:
        if not results:
            return {"average_score": 0.0, "graded": 0}

        total = 0.0
        component_totals = {"category": 0.0, "priority": 0.0, "routing": 0.0, "reply": 0.0}

        for r in results:
            reward = self.grade(r["email_id"], r["action"])
            total += reward.value
            for key in component_totals:
                component_totals[key] += reward.breakdown.get(key, 0.0)

        n = len(results)
        return {
            "average_score": round(total / n, 4),
            "total_score": round(total, 4),
            "avg_category": round(component_totals["category"] / n, 4),
            "avg_priority": round(component_totals["priority"] / n, 4),
            "avg_routing": round(component_totals["routing"] / n, 4),
            "avg_reply": round(component_totals["reply"] / n, 4),
            "graded": n,
        }

    def _score_priority(self, predicted: Priority, actual: Priority) -> float:
        order = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]
        diff = abs(order.index(predicted) - order.index(actual))
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5
        return 0.0

    def _score_reply(self, reply: str, ground_truth: dict) -> float:
        keywords = ground_truth.get("reply_keywords", [])

        # Spam — no reply expected
        if not keywords:
            return 0.0 if len(reply.strip()) > 10 else 1.0

        reply_lower = reply.lower()
        matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
        keyword_score = matched / len(keywords)

        # Length factor — reward replies of at least 30 words
        word_count = len(reply.split())
        length_factor = min(1.0, word_count / 30)

        return round((keyword_score * 0.7) + (length_factor * 0.3), 4)

    def _is_destructive(self, reply: str) -> bool:
        if not reply:
            return False
        bad_phrases = [
            "ignore", "delete this", "i don't care",
            "not my problem", "go away", "this is spam",
        ]
        return any(phrase in reply.lower() for phrase in bad_phrases)