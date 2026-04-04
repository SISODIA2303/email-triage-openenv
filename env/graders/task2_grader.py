from env.models import Action, Reward, Priority
from env.email_data import EMAIL_GROUND_TRUTH


class Task2Grader:
    """
    Task 2 - MEDIUM: Classification + Priority + Routing.
    Weights: category=0.40, priority=0.35, routing=0.25
    Priority has partial credit (off by one level = 0.5)
    """

    WEIGHTS = {
        "category": 0.40,
        "priority": 0.35,
        "routing": 0.25,
    }

    def grade(self, email_id: str, action: Action) -> Reward:
        if email_id not in EMAIL_GROUND_TRUTH:
            return Reward(
                value=0.0,
                breakdown={"category": 0.0, "priority": 0.0, "routing": 0.0},
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

        # --- Weighted total ---
        total = (
            category_score * self.WEIGHTS["category"] +
            priority_score * self.WEIGHTS["priority"] +
            routing_score * self.WEIGHTS["routing"]
        )

        return Reward(
            value=round(total, 4),
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts),
        )

    def grade_episode(self, results: list) -> dict:
        if not results:
            return {"total_score": 0.0, "average_score": 0.0, "graded": 0}

        total = 0.0
        category_correct = 0
        priority_correct = 0
        routing_correct = 0

        for r in results:
            reward = self.grade(r["email_id"], r["action"])
            total += reward.value
            if reward.breakdown.get("category", 0) == 1.0:
                category_correct += 1
            if reward.breakdown.get("priority", 0) == 1.0:
                priority_correct += 1
            if reward.breakdown.get("routing", 0) == 1.0:
                routing_correct += 1

        n = len(results)
        return {
            "total_score": round(total, 4),
            "average_score": round(total / n, 4),
            "category_accuracy": round(category_correct / n, 4),
            "priority_accuracy": round(priority_correct / n, 4),
            "routing_accuracy": round(routing_correct / n, 4),
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