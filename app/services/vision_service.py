from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os

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