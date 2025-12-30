from fastapi import APIRouter, HTTPException
from app.models.schemas import SimulationRequest, SimulationResponse
from app.services.simulation_service import generate_simulation_scenario
from app.services.lesson_service import get_lesson_word 

router = APIRouter()

@router.post("/simulation", response_model=SimulationResponse)
async def create_simulation(req: SimulationRequest):
    """
    [POST] /api/simulation
    오늘 배운 Lesson ID 리스트를 받아 시뮬레이션 생성
    """
    try:
        # 1. Lesson ID로 실제 백엔드에서 단어 이름 조회
        lesson_words = {}
        
        print(f"🔍 시뮬레이션 요청 수신: ID 목록 {req.lesson_ids}")

        for lid in req.lesson_ids:
            # 실제 API 호출
            word = get_lesson_word(lid)
            
            if word:
                lesson_words[lid] = word
            else:
                print(f"⚠️ 경고: 레슨 ID {lid}에 해당하는 단어를 가져오지 못했습니다. 시뮬레이션에서 제외됩니다.")
        
        # 조회된 단어가 하나도 없으면 에러 처리
        if not lesson_words:
            raise HTTPException(status_code=400, detail="유효한 레슨 단어를 찾을 수 없습니다. (DB 조회 실패 또는 잘못된 ID)")

        print(f"🤖 AI 생성 시작 (사용 단어: {lesson_words})")

        # 2. AI 서비스 호출 (기존 로직 동일)
        result = generate_simulation_scenario(lesson_words)
        
        return result

    except Exception as e:
        print(f"❌ 시뮬레이션 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))