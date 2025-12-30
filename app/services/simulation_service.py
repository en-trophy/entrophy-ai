import json
from app.ai.llm_client import call_gpt_json, call_dalle_image
from app.models.schemas import SimulationResponse, DialogueLine

def generate_simulation_scenario(lesson_words: dict) -> SimulationResponse:
    """
    lesson_words: {lesson_id: "단어명", ...}
    """
    words_str = ", ".join(lesson_words.values())
    
    # 1. GPT 프롬프트 작성
    system_prompt = """
    너는 수어를 가르치는 교육 앱의 시나리오 작가야.
    사용자가 오늘 배운 어휘들을 사용하여 대화하는 '1인칭 시점(AI 상대방이 사용자에게 말을 거는 형태)'의 짧은 상황극을 만들어 줘 

    [규칙]
    1. 대화는 반드시 AI(상대방)가 먼저 시작하고, AI가 마무리하며 끝나야 하며, 대화가 매우 자연스러워야 합니다.
    2. 'User'의 차례에는 사용자가 배워야 할 어휘 중 하나를 사용해서 가공 없이 해당 어휘로만 대답해야 합니다.
    3. 모든 배운 어휘가 한 번씩만 꼭 사용되어야 합니다. 즉 AI와 상대방의 대화는 (어휘 개수 * 2) + 1개 여야 합니다.
    4. 상황 설명은 DALL-E 3로 이미지를 생성하기 좋게 구체적이고 묘사적으로 작성하세요.
    5. 모든 텍스트는 영어로 작성합니다.

    [JSON 출력 포맷]
    {
        "situation": "상황 설명 (한글)",
        "image_prompt": "DALL-E를 위한 영어 이미지 프롬프트 (예: First-person view of...)",
        "dialogue": [
            { "speaker": "AI", "text": "..." },
            { "speaker": "User", "text": "...", "target_word": "어휘" }
        ]
    }
    """
    
    user_prompt = f"사용할 단어 목록: {json.dumps(lesson_words, ensure_ascii=False)}"

    # 2. GPT 호출 (JSON 모드)
    scenario_data = call_gpt_json(system_prompt, user_prompt)
    
    # 3. DALL-E 3 이미지 생성 호출
    # 프롬프트에 '1인칭 시점' 등 스타일 추가
    final_image_prompt = f"First-person view, photorealistic, {scenario_data['image_prompt']}, high quality"
    image_url = call_dalle_image(final_image_prompt)

    # 4. 결과 매핑 (단어명 -> ID 변환)
    word_to_id = {v: k for k, v in lesson_words.items()}
    dialogue_list = []
    
    for line in scenario_data["dialogue"]:
        lid = None
        # User 차례이고 target_word가 있으면 ID 찾기
        if line["speaker"] == "User" and "target_word" in line:
            target_w = line["target_word"]
            # 정확히 일치하거나 포함되는 단어 찾기
            for w, i in word_to_id.items():
                if w in target_w or target_w in w:
                    lid = i
                    break
        
        dialogue_list.append(DialogueLine(
            speaker=line["speaker"],
            text=line["text"],
            target_lesson_id=lid
        ))

    return SimulationResponse(
        situation=scenario_data["situation"],
        image_url=image_url,
        dialogue=dialogue_list
    )