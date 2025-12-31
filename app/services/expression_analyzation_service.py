from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
import base64
import requests

# 환경 변수에서 가져오기 (Azure Portal -> Configuration에 꼭 등록해야 함!)
ENDPOINT = os.environ.get("CUSTOM_VISION_ENDPOINT")
PREDICTION_KEY = os.environ.get("CUSTOM_VISION_KEY")
PROJECT_ID = os.environ.get("CUSTOM_VISION_PROJECT_ID")
PUBLISH_ITERATION_NAME = "face-expression-1" # Publish 할 때 지은 이름

# 클라이언트 초기화
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(ENDPOINT, credentials)

def analyze_expression(image_bytes: bytes) -> str:
    """
    이미지를 Custom Vision에 보내서 가장 확률 높은 표정 태그(문자열)를 반환
    """
    try:
        results = predictor.classify_image(
            PROJECT_ID, 
            PUBLISH_ITERATION_NAME, 
            image_bytes
        )
        
        # 확률(Probability)이 가장 높은 예측 결과 가져오기
        if not results.predictions:
            return "Unknown"

        best_prediction = max(results.predictions, key=lambda x: x.probability)
        
        # 확률이 너무 낮으면(예: 50% 미만) 신뢰할 수 없음 처리
        if best_prediction.probability < 0.5:
            return "Uncertain"
            
        return best_prediction.tag_name

    except Exception as e:
        print(f"❌ Custom Vision Error: {e}")
        return "Error"

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")


def analyze_expression_with_llm(image_bytes: bytes) -> str:
    """
    Azure OpenAI (gpt-4o-mini, Vision)으로
    표정을 Curious / Positive / Negative / Neutral 중 하나로 분류
    """

    try:
        # 이미지 → base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that analyzes a user's facial expression "
                        "from an image.\n"
                        "You MUST choose exactly one of the following labels:\n"
                        "- Curious (questioning, interested, wondering)\n"
                        "- Positive (happy, pleased, friendly)\n"
                        "- Negative (angry, sad, frustrated)\n"
                        "- Neutral (no clear emotion)\n\n"
                        "Respond with ONLY the label name."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify the facial expression in this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        label = result["choices"][0]["message"]["content"].strip()

        allowed = {"Curious", "Positive", "Negative", "Neutral"}
        return label if label in allowed else "Uncertain"

    except Exception as e:
        print(f"❌ LLM Expression Error: {e}")
        return "Error"
