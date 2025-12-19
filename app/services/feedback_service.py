from app.ai.llm_client import call_llm

def generate_feedback(lesson_id, user_feature, answer_feature, evaluation):

    prompt = f"""
            너는 미국 수어 학습 코치야.

            [정답 수어 특징]
            {answer_feature}

            [사용자 수어 특징]
            {user_feature}

            [판정]
            정답 여부: {evaluation['is_correct']}
            점수: {evaluation['score']}

            사용자에게 짧고 친절하게 영어로 피드백해 줘.
            틀렸다면 어떤 부분을 고치면 좋은지 말해줘.
            """

    return call_llm(prompt)
