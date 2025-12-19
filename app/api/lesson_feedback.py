from fastapi import APIRouter
from app.models.schemas import LessonFeedbackRequest, LessonFeedbackResponse
from app.services.feature_extractor import extract_feature_json2
from app.services.lesson_service import get_answer_frame, get_test_answer
from app.services.evaluation_service import evaluate_static_sign
from app.services.feedback_service import generate_feedback

router = APIRouter()

@router.post("/{lessonId}/feedback", response_model=LessonFeedbackResponse)
async def lesson_feedback(lessonId: int, req: LessonFeedbackRequest):

    # 1. raw landmarks → feature json
    user_feature = extract_feature_json2(req.raw_landmarks)

    # 2. 정답 frame 조회
    answer_feature = get_test_answer()
    # answer_feature = get_answer_frame(lessonId)

    # 3. 정답 여부 판단
    result = evaluate_static_sign(user_feature, answer_feature)

    # 4. 자연어 피드백 생성
    feedback = generate_feedback(
        lesson_id=lessonId,
        user_feature=user_feature,
        answer_feature=answer_feature,
        evaluation=result
    )

    return LessonFeedbackResponse(
        isCorrect=result["is_correct"],
        score=result["score"],
        feedback=feedback
    )
