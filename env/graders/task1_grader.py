from env.models import Action, Reward
from env.email_data import EMAIL_GROUND_TRUTH


class Task1Grader:
    """
    Task 1 - EASY: Email Classification only.
    Score strictly between 0.001 and 0.999.
    """

    def grade(self, email_id: str, action: Action) -> Reward:
        if email_id not in EMAIL_GROUND_TRUTH:
            return Reward(
                value=0.001,
                breakdown={"category": 0.001},
                feedback=f"Unknown email_id: {email_id}",
            )

        ground_truth = EMAIL_GROUND_TRUTH[email_id]
        expected_category = ground_truth["category"]
        correct = action.category == expected_category
        score = 0.999 if correct else 0.001

        feedback = (
            f"Category: correct ({action.category.value})"
            if correct
            else f"Category: got '{action.category.value}', expected '{expected_category.value}'"
        )

        return Reward(
            value=round(score, 4),
            breakdown={"category": score},
            feedback=feedback,
        )

    def grade_episode(self, results: list) -> dict:
        if not results:
            return {"total_score": 0.001, "accuracy": 0.001, "graded": 0}

        total = 0.0
        correct_count = 0

        for r in results:
            reward = self.grade(r["email_id"], r["action"])
            total += reward.value
            if reward.value >= 0.999:
                correct_count += 1

        n = len(results)
        accuracy = correct_count / n
        accuracy = max(0.001, min(0.999, accuracy))

        return {
            "total_score": round(total, 4),
            "average_score": round(max(0.001, min(0.999, total / n)), 4),
            "accuracy": round(accuracy, 4),
            "correct": correct_count,
            "total": n,
            "graded": n,
        }