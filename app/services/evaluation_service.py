from app.utils.similarity import compare_feature
import json

def evaluate_static_sign(user: dict, answer: dict) -> dict:
    score = compare_feature(user, answer)
    return {
        "score": score,
        "is_correct": score == 1.0
    }
