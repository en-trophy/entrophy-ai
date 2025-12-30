import os
import requests
import json

# 환경 변수 로드
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

# 배포 이름 (기존 GPT용 + 새로 만든 DALL-E용)
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini") # 기존꺼 재사용
AZURE_DALLE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DALLE_DEPLOYMENT", "dalle-3") # [NEW] 새로 만든 배포 이름

def get_headers():
    return {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

# [기존 함수 유지] 피드백 생성용
def call_llm(prompt: str) -> str:
    # ... (기존 코드와 동일하게 유지하거나, 아래 call_gpt_generic을 호출하도록 리팩토링 해도 됨)
    # 님 코드 그대로 두셔도 됩니다.
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    payload = {
        "messages": [
            {"role": "system", "content": "너는 미국 수어(KSL) 학습자를 돕는... (생략)"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 300
    }
    response = requests.post(url, headers=get_headers(), json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# [NEW] 시나리오 생성용 (JSON 모드 지원)
def call_gpt_json(system_prompt: str, user_prompt: str) -> dict:
    dalle_api_version = "2024-02-01"
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={dalle_api_version}"
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7, # 창의적인 시나리오를 위해 약간 높임
        "response_format": {"type": "json_object"} # JSON 강제 반환 설정
    }

    response = requests.post(url, headers=get_headers(), json=payload)
    response.raise_for_status()
    
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)

# [NEW] 이미지 생성용 (DALL-E 3)
def call_dalle_image(prompt: str) -> str:
    # DALL-E 3 URL 구조는 chat과 다름 (images/generations)
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_DALLE_DEPLOYMENT}/images/generations?api-version={API_VERSION}"
    
    payload = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "style": "vivid", # 선명한 스타일
        "quality": "standard"
    }

    response = requests.post(url, headers=get_headers(), json=payload)
    response.raise_for_status()
    
    # 이미지 URL 반환
    return response.json()["data"][0]["url"]