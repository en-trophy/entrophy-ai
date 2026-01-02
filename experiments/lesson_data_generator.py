import sys
import os
import cv2
import mediapipe as mp
import json
import time
import numpy as np
import requests
import tempfile
from datetime import datetime

# 현재 파일(answer_generator.py)의 부모의 부모 디렉토리(프로젝트 루트)를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from app.services.feature_extractor import extract_feature_json
from app.services.expression_analyzation_service import analyze_expression_with_llm

API_BASE_URL = os.getenv("BACKEND_ENDPOINT")
X_ADMIN_KEY = os.getenv("X_ADMIN_KEY")

# === 설정 및 초기화 ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def generate_static_lesson():
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        print(">>> 3초 카운트다운 시작!")
        start_time = time.time()
        
        captured_data = None
        final_image = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 분석용 원본 프레임 (정방향)
            analysis_frame = frame.copy()
            # 화면 출력용 프레임 (거울모드)
            display_frame = cv2.flip(frame, 1)

            elapsed = time.time() - start_time
            remaining = 3 - elapsed

            if remaining > 0:
                text = str(int(remaining) + 1)
                cv2.putText(display_frame, text, (300, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 255), 10)
                cv2.imshow('Hand Capture', display_frame)
                cv2.waitKey(1)
                continue
            else:
                print(">>> 캡처 및 분석 중...")

                # 1. MediaPipe 분석은 '정방향(analysis_frame)'으로 수행
                image = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                # LLM 분석용 바이트 변환 (정방향 사용)
                _, buffer = cv2.imencode('.jpg', analysis_frame)
                image_bytes = buffer.tobytes()

                expression = analyze_expression_with_llm(image_bytes)
                captured_data = extract_feature_json(results, expression)
                
                # 2. [수정됨] 업로드용 이미지는 '거울모드'로 저장
                final_image = cv2.flip(analysis_frame, 1) 

                cv2.putText(display_frame, "Captured!", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Hand Capture', display_frame)
                cv2.waitKey(1000)
                break

    cap.release()
    cv2.destroyAllWindows()
    return captured_data, final_image

def generate_dynamic_lesson(duration_sec):
    cap = cv2.VideoCapture(0)
    
    # 비디오 저장을 위한 설정
    temp_video_path = os.path.join(tempfile.gettempdir(), f"sign_video_{int(time.time())}.mp4")
    
    # 코덱 및 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    captured_jsons = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        print(">>> 3초 카운트다운 시작! (동영상 촬영)")
        start_time = time.time()
        while (time.time() - start_time) < 3:
            ret, frame = cap.read()
            if not ret: break
            display_frame = cv2.flip(frame, 1)
            remaining = 3 - (time.time() - start_time)
            cv2.putText(display_frame, str(int(remaining) + 1), (300, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 255), 10)
            cv2.imshow('Video Capture', display_frame)
            cv2.waitKey(1)

        print(">>> 녹화 시작!")
        record_start_time = time.time()
        last_capture_time = record_start_time
        capture_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            elapsed = current_time - record_start_time

            if elapsed > duration_sec:
                break

            # [수정됨] 비디오 파일에는 '거울모드(반전된 프레임)'를 기록
            out.write(cv2.flip(frame, 1))

            # 화면 표시 (거울모드)
            display_frame = cv2.flip(frame, 1)
            cv2.putText(display_frame, "REC", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Video Capture', display_frame)
            
            # 1초마다 프레임 캡처 및 분석
            if capture_count < duration_sec and (current_time - last_capture_time) >= 1.0:
                print(f"Analyzing frame {capture_count + 1}...")
                
                # 분석용 프레임은 '정방향(원본 frame)' 사용
                analysis_frame = frame.copy()
                image_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)

                # JPEG Bytes 변환 (분석용이므로 정방향)
                _, buffer = cv2.imencode('.jpg', analysis_frame)
                image_bytes = buffer.tobytes()

                expression = analyze_expression_with_llm(image_bytes)
                feature_json = extract_feature_json(results, expression)
                
                captured_jsons.append(feature_json)
                
                last_capture_time = current_time
                capture_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f">>> 녹화 완료: {temp_video_path}")
    print(f">>> 캡처된 프레임 수: {len(captured_jsons)}")
    
    return captured_jsons, temp_video_path

def post_images(image):
    url = f"{API_BASE_URL}/api/storage/images"
    
    # OpenCV 이미지를 바이트로 변환
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    # multipart/form-data 설정
    files = {
        'file': ('image.jpg', img_bytes, 'image/jpeg')
    }
    headers = {'X-ADMIN-KEY': X_ADMIN_KEY}

    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Image Upload Success: {result['uploadUrl']}")
        return result['uploadUrl']
    except Exception as e:
        print(f"❌ Image Upload Failed: {e}")
        return None

def post_videos(video_path):
    url = f"{API_BASE_URL}/api/storage/videos"
    
    headers = {'X-ADMIN-KEY': X_ADMIN_KEY}
    
    try:
        with open(video_path, 'rb') as f:
            files = {
                'file': ('video.mp4', f, 'video/mp4')
            }
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            result = response.json()
            print(f"✅ Video Upload Success: {result['uploadUrl']}")
            return result['uploadUrl']
    except Exception as e:
        print(f"❌ Video Upload Failed: {e}")
        return None

def post_lessons(category_id, title, sign_language, difficulty, type, mode, frame_number, image_url, video_url):
    url = f"{API_BASE_URL}/api/lessons"
    
    payload = {
        "categoryId": category_id,
        "title": title,
        "signLanguage": sign_language,
        "difficulty": difficulty,
        "type": type,
        "mode": mode,
        "imageUrl": image_url,
        "videoUrl": video_url,
        "frameNumber": frame_number
    }
    
    headers = {
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Lesson Created: ID {result['id']}")
        return result['id']
    except Exception as e:
        print(f"❌ Lesson Creation Failed: {e}")
        return None

def post_answer_frames(lesson_id, seq, answer_frame):
    url = f"{API_BASE_URL}/api/lessons/{lesson_id}/answer-frames"
    
    payload = {
        "seq": seq,
        "hand": answer_frame, 
        "frameMeta": "meta_data_placeholder"
    }
    
    json_data = json.dumps(payload, cls=NumpyEncoder)
    
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, data=json_data, headers=headers)
        response.raise_for_status()
        print(f"✅ Answer Frame {seq} Uploaded")
    except Exception as e:
        print(f"❌ Answer Frame Upload Failed: {e}")

def main():
    print("=== Sign Language Content Generator ===")
    try:
        category_id = int(input("Category ID : "))
        title = input("Title : ")
        sign_language = input("Sign Language (e.g. KSL, ASL) : ")
        difficulty = int(input("Difficulty (1-5) : "))
        is_word = input("Is Word? (y/n) : ").lower().startswith('y')
        type = "WORD" if is_word else "PHRASE"
        frame_number = int(input("Frame Number (Duration in sec) : "))
    except ValueError:
        print("Invalid Input.")
        return

    mode = None
    image_url = None
    video_url = None
    lesson_id = None

    if frame_number == 1:
        # 정적 이미지 (단어)
        mode = "STATIC"
        hand_json, image = generate_static_lesson()
        if image is not None:
            image_url = post_images(image)
            if image_url:
                lesson_id = post_lessons(category_id, title, sign_language, difficulty, type, mode, frame_number, image_url, video_url)
                if lesson_id:
                    post_answer_frames(lesson_id, 1, hand_json)
    
    else:
        # 동적 비디오 (문장 또는 긴 단어)
        mode = "DYNAMIC"
        hand_jsons, video_path = generate_dynamic_lesson(frame_number)
        
        if video_path and os.path.exists(video_path):
            video_url = post_videos(video_path)
            
            if video_url:
                lesson_id = post_lessons(category_id, title, sign_language, difficulty, type, mode, frame_number, image_url, video_url)
                
                if lesson_id:
                    for i, hand_json in enumerate(hand_jsons):
                        post_answer_frames(lesson_id, i + 1, hand_json)

if __name__ == "__main__":
    main()