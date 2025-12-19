import json
from pathlib import Path

def get_answer_frame(lessonId: int) -> dict:
    """
    GET /api/lessons/{lessonId}/answer-frames
    에서 가져온 JSON이라고 가정
    """

    # TODO: 실제로는 DB / API 호출
    return {
        "left": {...},
        "right": {...},
        "inter_hand_relation": {...},
        "finger_relation": {...}
    }


def get_test_answer():
    """
    임시 테스트용 정답 로더
    answers/hand.json 파일을 정답으로 간주한다.
    """

    # 현재 파일 기준 경로
    base_dir = Path(__file__).resolve().parent.parent
    answer_path = base_dir / "answers" / "hand.json"

    if not answer_path.exists():
        raise FileNotFoundError(f"정답 파일을 찾을 수 없습니다: {answer_path}")

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    return answer_data
