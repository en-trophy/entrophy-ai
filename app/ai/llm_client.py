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
    print("ENDPOINT =", AZURE_OPENAI_ENDPOINT)
    print("DEPLOYMENT =", AZURE_OPENAI_DEPLOYMENT)
    print("API KEY EXISTS =", bool(AZURE_OPENAI_API_KEY))

    url = (
        f"{AZURE_OPENAI_ENDPOINT}"
        f"openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={API_VERSION}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 미국 수어(KSL) 학습자를 돕는 친절한 수어 코치야. "
                    "정답이면 짧게 축하만 해 주고, 틀렸으면 [정답 수어 특징]과 [사용자 수어 특징]을 비교해서 사용자가 왜 틀렸는지 핵심만 사용자에게 아주 짧게 영어 1 문장으로 피드백해 줘. 표정이 틀렸으면 그것도 포함해서 알려줘. 강조 기호는 사용하지마."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4,
        "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]

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