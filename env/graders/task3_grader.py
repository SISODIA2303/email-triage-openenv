from env.models import Action, Reward, Priority
from env.email_data import EMAIL_GROUND_TRUTH


class Task3Grader:
    """
    Task 3 - HARD: Full triage pipeline.
    Weights: category=0.25, priority=0.25, routing=0.20, reply=0.30
    Score strictly between 0.001 and 0.999.
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
                value=0.001,
                breakdown={"category": 0.001, "priority": 0.001, "routing": 0.001, "reply": 0.001},
                feedback=f"Unknown email_id: {email_id}",
            )

        ground_truth = EMAIL_GROUND_TRUTH[email_id]
        breakdown = {}
        feedback_parts = []

        # --- Category ---
        category_correct = action.category == ground_truth["category"]
        category_score = 0.999 if category_correct else 0.001
        breakdown["category"] = category_score
        expected_cat = ground_truth["category"].value
        feedback_parts.append(
            f"Category: correct ({action.category.value})"
            if category_correct
            else f"Category: got '{action.category.value}' expected '{expected_cat}'"
        )

        # --- Priority ---
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

        # --- Routing ---
        if action.routing_team is None:
            routing_score = 0.001
            feedback_parts.append("Routing: missing")
        else:
            routing_correct = action.routing_team == ground_truth["routing_team"]
            routing_score = 0.999 if routing_correct else 0.001
            expected_route = ground_truth["routing_team"].value
            feedback_parts.append(
                f"Routing: correct ({action.routing_team.value})"
                if routing_correct
                else f"Routing: got '{action.routing_team.value}' expected '{expected_route}'"
            )
        breakdown["routing"] = round(routing_score, 4)

        # --- Reply ---
        if action.reply_draft is None or action.reply_draft.strip() == "":
            reply_score = 0.001
            feedback_parts.append("Reply: missing")
        else:
            reply_score = self._score_reply(action.reply_draft, ground_truth)
            reply_score = max(0.001, min(0.999, reply_score))
            feedback_parts.append(f"Reply: {round(reply_score, 2)} score")
        breakdown["reply"] = round(reply_score, 4)

        # --- Penalty ---
        penalty = 0.0
        if self._is_destructive(action.reply_draft):
            penalty = 0.3
            feedback_parts.append("penalty: destructive reply")

        total = (
            category_score * self.WEIGHTS["category"] +
            priority_score * self.WEIGHTS["priority"] +
            routing_score * self.WEIGHTS["routing"] +
            reply_score * self.WEIGHTS["reply"]
        )
        total = max(0.0, total - penalty)
        total = max(0.001, min(0.999, round(total, 4)))

        return Reward(
            value=total,
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts),
        )

    def grade_episode(self, results: list) -> dict:
        if not results:
            return {"average_score": 0.001, "graded": 0}

        total = 0.0
        component_totals = {"category": 0.0, "priority": 0.0, "routing": 0.0, "reply": 0.0}

        for r in results:
            reward = self.grade(r["email_id"], r["action"])
            total += reward.value
            for key in component_totals:
                component_totals[key] += reward.breakdown.get(key, 0.0)

        n = len(results)
        return {
            "average_score": round(max(0.001, min(0.999, total / n)), 4),
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

    def _is_destructive(self, reply: str) -> bool:
        if not reply:
            return False
        bad_phrases = [
            "ignore", "delete this", "i don't care",
            "not my problem", "go away", "this is spam",
        ]
        return any(phrase in reply.lower() for phrase in bad_phrases)