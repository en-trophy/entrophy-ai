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

        정답이면 짧게 축하만 해 주고, 틀렸으면 [정답 수어 특징]과 [사용자 수어 특징]을 비교해서 사용자가 왜 틀렸는지만
            사용자에게 아주 짧고 핵심적으로 1 문장의 영어로 피드백해 줘. 표정이 틀렸으면 그것도 같이 말해줘. 강조 기호는 사용하지마. e.g.) Stretch your pinky finger, turn your wrist forward, etc..
            """

    return call_llm(prompt)
