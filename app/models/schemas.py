from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 랜드마크 점 하나 (x, y, z, visibility)
class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None

# 2. 4가지 부위별 데이터 (MediaPipe Holistic 구조)
class HolisticData(BaseModel):
    # 프런트에서 감지 안 된 부위는 null로 보낼 것이므로 Optional 필수
    face_landmarks: Optional[List[Landmark]] = None
    pose_landmarks: Optional[List[Landmark]] = None
    left_hand_landmarks: Optional[List[Landmark]] = None
    right_hand_landmarks: Optional[List[Landmark]] = None

class DummyResults:
    def __init__(self, data):
        # 1. 들어온 데이터가 Pydantic 모델이면 딕셔너리로 강제 변환
        if hasattr(data, "model_dump"):
            self._data = data.model_dump()
        elif hasattr(data, "dict"):
            self._data = data.dict()
        elif isinstance(data, dict):
            self._data = data
        else:
            self._data = {}

        # 2. 데이터 파싱 (키 이름이 달라도 알아서 찾음)
        self.left_hand_landmarks = self._parse_landmarks(['leftHand', 'left_hand', 'left_hand_landmarks'])
        self.right_hand_landmarks = self._parse_landmarks(['rightHand', 'right_hand', 'right_hand_landmarks'])
        self.pose_landmarks = self._parse_landmarks(['pose', 'pose_landmarks'])
        self.face_landmarks = self._parse_landmarks(['face', 'face_landmarks'])

    def _parse_landmarks(self, keys):
        for k in keys:
            items = self._data.get(k)
            if items:
                landmarks = []
                for p in items:
                    # p가 딕셔너리인 경우
                    if isinstance(p, dict):
                        landmarks.append(Landmark(p.get('x', 0), p.get('y', 0), p.get('z', 0), p.get('visibility', 0)))
                    # p가 객체인 경우
                    elif hasattr(p, 'x'):
                        landmarks.append(Landmark(p.x, p.y, p.z, getattr(p, 'visibility', 0)))
                return landmarks
        return None

    # ✅ 핵심: 이 메서드가 없어서 에러가 났던 것입니다. 반드시 포함되어야 합니다.
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    # 추가: 딕셔너리처럼 ['key']로 접근해도 안 터지게 방어
    def __getitem__(self, key):
        return self._data.get(key)
# ==========================================

# 3. 최종 요청 바디 (Request Body)
class LessonFeedbackRequest(BaseModel):
    target_word_id: int        # 정답 단어 ID
    raw_landmarks: HolisticData # 위에서 정의한 랜드마크 묶음

class LessonFeedbackResponse(BaseModel):
    isCorrect: bool
    score: float
    feedback: str
