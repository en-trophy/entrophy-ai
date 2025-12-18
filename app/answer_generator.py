
import json
import mediapipe as mp
from datetime import datetime
import time
import cv2
import phonoqy_extractor
import numpy as np

def normalize_json_types(obj):
    if isinstance(obj, dict):
        return {k: normalize_json_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_json_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def main():

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # 손 인식 모델 초기화
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 시작 시간 기록
    start_time = time.time()
    countdown_seconds = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 거울 모드 (좌우 반전) - 보기에 더 편함
        frame = cv2.flip(frame, 1)

        # 이미지 변환 (BGR -> RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 손 인식 수행
        results = hands.process(image)
        
        # 그리기 모드로 복구
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기 (화면 확인용)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === 카운트다운 로직 ===
        elapsed_time = time.time() - start_time
        remaining_time = countdown_seconds - elapsed_time

        if remaining_time > 0:
            # 화면 중앙에 크게 카운트다운 숫자 표시
            text = f"{int(remaining_time) + 1}"
            font_scale = 10
            thickness = 20
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (image.shape[0] + text_size[1]) // 2
            
            # 텍스트 외곽선 (검정)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness + 10)
            # 텍스트 본체 (노랑)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 255, 255), thickness)
            
            cv2.imshow('Hand Capture Countdown', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            if not results.multi_hand_landmarks:
                print("손이 감지되지 않았습니다.")
            
            print("="*50)
            print("카메라를 종료합니다.")

            hand_json = {
                "left": {"present": False},
                "right": {"present": False}
            }

            if results.multi_hand_landmarks:
                for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    side = handed.classification[0].label.lower()  # 'left' / 'right'
                    h, w, _ = frame.shape
                    hand_json[side] = phonoqy_extractor.extract_hand_features(lm, w, h)
                    hand_json[side]["present"] = True
            
            # 잠시 캡처된 화면(마지막 프레임)을 1초간 보여주고 꺼짐
            cv2.imshow('Hand Capture Countdown', image)
            cv2.waitKey(1000) 
            break

    cap.release()
    cv2.destroyAllWindows()

    filename = datetime.now().strftime("%Y%m%d%H%M") + ".json"

    with open(filename, "w", encoding="utf-8") as f:
        hand_json = normalize_json_types(hand_json)
        json.dump(hand_json, f, indent=2)

    print(f"Saved: {filename}")

main()