from env.models import Action, Reward, Priority
from env.email_data import EMAIL_GROUND_TRUTH


class Task2Grader:
    """
    Task 2 - MEDIUM: Classification + Priority + Routing.
    Weights: category=0.40, priority=0.35, routing=0.25
    Score strictly between 0.001 and 0.999.
    """

    WEIGHTS = {
        "category": 0.40,
        "priority": 0.35,
        "routing": 0.25,
    }

    def grade(self, email_id: str, action: Action) -> Reward:
        if email_id not in EMAIL_GROUND_TRUTH:
            return Reward(
                value=0.001,
                breakdown={"category": 0.001, "priority": 0.001, "routing": 0.001},
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

        # --- Weighted total ---
        total = (
            category_score * self.WEIGHTS["category"] +
            priority_score * self.WEIGHTS["priority"] +
            routing_score * self.WEIGHTS["routing"]
        )
        total = max(0.001, min(0.999, round(total, 4)))

        return Reward(
            value=total,
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts),
        )

    def grade_episode(self, results: list) -> dict:
        if not results:
            return {"total_score": 0.001, "average_score": 0.001, "graded": 0}

        total = 0.0
        category_correct = 0
        priority_correct = 0
        routing_correct = 0

        for r in results:
            reward = self.grade(r["email_id"], r["action"])
            total += reward.value
            if reward.breakdown.get("category", 0) >= 0.999:
                category_correct += 1
            if reward.breakdown.get("priority", 0) >= 0.999:
                priority_correct += 1
            if reward.breakdown.get("routing", 0) >= 0.999:
                routing_correct += 1

        n = len(results)
        return {
            "total_score": round(total, 4),
            "average_score": round(max(0.001, min(0.999, total / n)), 4),
            "category_accuracy": round(max(0.001, min(0.999, category_correct / n)), 4),
            "priority_accuracy": round(max(0.001, min(0.999, priority_correct / n)), 4),
            "routing_accuracy": round(max(0.001, min(0.999, routing_correct / n)), 4),
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