import json
from pathlib import Path

import requests
import json

# 실제 백엔드 서버 주소로 변경해줘 (예: http://localhost:8080)
API_BASE_URL = "https://equal-sign-backend-api-haejb5bdhnezc2c2.koreacentral-01.azurewebsites.net" 

def get_answer_frame(lessonId: int) -> dict:
    """
    GET /api/lessons/{lessonId}/answer-frames
    API 응답의 'hand' 필드 안에 있는 데이터를 추출하여 반환
    """
    url = f"{API_BASE_URL}/api/lessons/{lessonId}/answer-frames"
    
    # 반환할 기본 구조 초기화
    result_data = {
        "left": {},
        "right": {},
        "inter_hand_relation": {},
        "finger_relation": {}
    }

    try:
        response = requests.get(url)
        response.raise_for_status()
        
        frames_list = response.json()
        
        # 데이터가 비어있으면 빈 딕셔너리 반환
        if not frames_list:
            print("⚠️ API 응답 리스트가 비어있습니다.")
            return result_data

        # 리스트의 첫 번째 아이템(혹은 필요한 아이템)을 가져옵니다.
        # 보통 정답 프레임은 하나라고 가정합니다.
        item = frames_list[0]
        
        # 'hand' 딕셔너리를 가져옴 (여기에 진짜 데이터가 있음)
        hand_data = item.get('hand', {})
        
        if not isinstance(hand_data, dict):
            print(f"⚠️ 'hand' 필드가 딕셔너리가 아닙니다: {type(hand_data)}")
            return result_data

        # API 구조 그대로 매핑
        # hand_data 안에 이미 'left', 'right' 키가 존재함
        if 'left' in hand_data:
            result_data['left'] = hand_data['left']
        
        if 'right' in hand_data:
            result_data['right'] = hand_data['right']
            
        if 'inter_hand_relation' in hand_data:
            result_data['inter_hand_relation'] = hand_data['inter_hand_relation']
            
        if 'finger_relation' in hand_data:
            result_data['finger_relation'] = hand_data['finger_relation']

        print("✅ 정답 데이터 로딩 성공!")
        return result_data

    except requests.exceptions.RequestException as e:
        print(f"❌ API 호출 중 오류 발생: {e}")
        return result_data
    

def get_test_answer_frame():
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
