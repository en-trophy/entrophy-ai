import cv2
import mediapipe as mp
import json
import time
import numpy as np
from phonoqy_extractor import extract_phonogy_json

# === 설정 및 초기화 ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# [수정됨] JSON 저장 시 numpy 타입을 Python 기본 타입으로 변환해주는 클래스
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():

    word = input("Word : ")

    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        print(">>> 3초 카운트다운 시작!")
        start_time = time.time()
        
        captured_data = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            
            elapsed = time.time() - start_time
            remaining = 3 - elapsed
            
            if remaining > 0:
                text = str(int(remaining) + 1)
                cv2.putText(frame, text, (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                            7, (0, 255, 255), 10)
                cv2.imshow('Hand Capture', frame)
                cv2.waitKey(1)
                continue
                
            else:
                print(">>> 캡처 및 분석 중...")
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                # 정답 json 추출
                captured_data = extract_phonogy_json(results)
                
                cv2.putText(frame, "Captured!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (0, 255, 0), 3)
                cv2.imshow('Hand Capture', frame)
                cv2.waitKey(1000)
                break
        
    cap.release()
    cv2.destroyAllWindows()

    if captured_data:
        filename = word + ".json"
        with open(filename, 'w', encoding='utf-8') as f:
            # [수정됨] cls=NumpyEncoder 추가하여 에러 해결
            json.dump(captured_data, f, indent=2, cls=NumpyEncoder)
        print(f"\n✅ 성공적으로 저장되었습니다: {filename}")

if __name__ == "__main__":
    main()