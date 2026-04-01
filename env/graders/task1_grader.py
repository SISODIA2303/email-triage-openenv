from env.models import Action, Reward
from env.email_data import EMAIL_GROUND_TRUTH


class Task1Grader:
    """
    Task 1 - EASY: Email Classification only.
    Agent must correctly classify the email category.
    Score: 1.0 = correct, 0.0 = wrong. No partial credit.
    """

    def grade(self, email_id: str, action: Action) -> Reward:
        if email_id not in EMAIL_GROUND_TRUTH:
            return Reward(
                value=0.0,
                breakdown={"category": 0.0},
                feedback=f"Unknown email_id: {email_id}",
            )

        ground_truth = EMAIL_GROUND_TRUTH[email_id]
        expected_category = ground_truth["category"]

        correct = action.category == expected_category
        score = 1.0 if correct else 0.0

        feedback = (
            f"Category: ✓ correct ({action.category.value})"
            if correct
            else f"Category: ✗ got '{action.category.value}', expected '{expected_category.value}'"
        )

        return Reward(
            value=round(score, 4),
            breakdown={"category": score},
            feedback=feedback,
        )

    def grade_episode(self, results: list) -> dict:
        """
        Grade a full episode of results.
        results = list of dicts with keys: email_id, action
        Returns summary statistics.
        """
        if not results:
            return {"total_score": 0.0, "accuracy": 0.0, "graded": 0}

        total = 0.0
        correct_count = 0

        for r in results:
            reward = self.grade(r["email_id"], r["action"])
            total += reward.value
            if reward.value == 1.0:
                correct_count += 1

        accuracy = correct_count / len(results)

        return {
            "total_score": round(total, 4),
            "average_score": round(total / len(results), 4),
            "accuracy": round(accuracy, 4),
            "correct": correct_count,
            "total": len(results),
            "graded": len(results),
        }