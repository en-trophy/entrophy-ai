from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import LessonFeedbackRequest, LessonFeedbackResponse
from app.services.feature_extractor import extract_feature_json
from app.services.lesson_service import get_answer_frame, get_test_answer_frame
from app.services.evaluation_service import evaluate_static_sign
from app.services.feedback_service import generate_feedback
from app.utils.mediapipe_adapter import build_mediapipe_results_from_request
from app.services.mediapipe_service import process_image_to_landmarks

router = APIRouter()

@router.post("/{lessonId}/feedback", response_model=LessonFeedbackResponse)
async def lesson_feedback(lessonId: int, req: LessonFeedbackRequest):

    results = build_mediapipe_results_from_request(req.raw_landmarks)

    # 1. raw landmarks → feature json
    user_feature = extract_feature_json(results)

    # 2. 정답 frame 조회
    # answer_feature = get_test_answer()
    answer_feature = get_answer_frame(lessonId)

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


@router.post("/{lessonId}/feedback/image", response_model=LessonFeedbackResponse)
async def lesson_feedback_by_image(
    lessonId: int, 
    file: UploadFile = File(...)
):
    try:
        # 1. 이미지 읽기
        image_bytes = await file.read()
        
        # 2. 이미지 -> MediaPipe 추론 -> DummyResults 변환 (어댑터 사용)
        results = process_image_to_landmarks(image_bytes)

        # 3. raw landmarks → feature json (기존 로직 재사용)
        user_feature = extract_feature_json(results)

        # 4. 정답 frame 조회 (DB/API)
        answer_feature = get_answer_frame(lessonId)

        # 5. 정답 여부 판단
        result = evaluate_static_sign(user_feature, answer_feature)

        # 6. 자연어 피드백 생성
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

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error processing image feedback: {e}")
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")