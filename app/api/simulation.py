from fastapi import APIRouter, HTTPException
from app.models.schemas import SimulationRequest, SimulationResponse
from app.services.simulation_service import generate_simulation_scenario
# from app.services.lesson_service import get_word_by_id # (구현 필요)

router = APIRouter()

@router.post("/simulation", response_model=SimulationResponse)
async def create_simulation(req: SimulationRequest):
    """
    [POST] /api/simulation
    오늘 배운 Lesson ID 리스트를 받아 시뮬레이션 생성
    """
    try:
        # 1. Lesson ID로 단어 이름 조회 (DB 연동)
        # 예시: {1: "고구마", 2: "사랑해"}
        lesson_words = {}
        
        # [TODO] 실제 DB 조회 로직으로 교체 필요
        # for lid in req.lesson_ids:
        #     word = get_word_by_id(lid)
        #     if word: lesson_words[lid] = word
        
        # 테스트용 목업 데이터 (DB 연결 전 테스트 시 사용)
        mock_db = {1: "SWEET POTATO", 2: "I LOVE YOU"}
        for lid in req.lesson_ids:
            if lid in mock_db:
                lesson_words[lid] = mock_db[lid]
        
        if not lesson_words:
            raise HTTPException(status_code=400, detail="유효한 레슨 ID가 없습니다.")

        # 2. AI 서비스 호출
        result = generate_simulation_scenario(lesson_words)
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))