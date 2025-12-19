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

         정답이 아니면 [정답 수어 특징]과 [사용자 수어 특징]을 비교해서 어떤 부분을 고쳐야 할지
            사용자에게 짧고 친절하게 1-2문장 이내로 피드백해 줘.
            """

    return call_llm(prompt)
