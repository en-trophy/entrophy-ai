import os
import requests

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

def call_llm(prompt: str) -> str:
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
                    "전문적이고 따뜻한 말투의 영어로 정답 정보와 사용자 답 정보를 비교해서 만약 정답이 아니면 아주 간단히 왜 틀렸는지 설명해줘."
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
